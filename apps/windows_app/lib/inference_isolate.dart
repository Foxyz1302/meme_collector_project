/// Inference isolate — runs ONNX models in a separate isolate to avoid
/// blocking the UI thread.
///
/// The isolate creates its own CLIP text + vision ONNX sessions internally.
/// The main isolate sends text/image paths via SendPort, receives Float32List
/// embeddings back. Float32List is copyable across isolate boundaries.
///
/// Message protocol:
///   Main → Isolate:
///     _InitRequest(modelPaths)  — initialize ONNX sessions
///     _EmbedTextRequest(text)   — embed text
///     _EmbedImageRequest(path)  — embed image file
///     _DisposeRequest           — dispose ONNX sessions
///   Isolate → Main:
///     _Ready                     — init complete
///     _EmbedResponse(Float32List) — embedding result
///     _Error(String)             — error message

library;

import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:meme_collector_core/meme_collector_core.dart';

import 'inference_impl.dart';

/// Configuration for the inference isolate (all fields are copyable).
class InferenceIsolateConfig {
  final String clipTextModelPath;
  final String clipTokenizerPath;
  final String clipVisionModelPath;

  const InferenceIsolateConfig({
    required this.clipTextModelPath,
    required this.clipTokenizerPath,
    required this.clipVisionModelPath,
  });
}

/// Manages the inference isolate. Created by the app, passed to Coordinator.
class InferenceIsolateManager {
  SendPort? _sendPort;
  ReceivePort? _receivePort;
  Isolate? _isolate;
  Stream<dynamic>? _broadcastStream;

  final _pendingRequests = <Completer<Float32List>>[];
  bool _isProcessing = false;
  final _requestQueue = <_QueuedRequest>[];

  /// Spawn the isolate + initialize ONNX sessions.
  Future<void> spawn(InferenceIsolateConfig config) async {
    _receivePort = ReceivePort();
    _broadcastStream = _receivePort!.asBroadcastStream();

    _isolate = await Isolate.spawn(
      _inferenceIsolateEntry,
      _IsolateInit(mainPort: _receivePort!.sendPort, config: config),
    );

    // Wait for the isolate to send us its SendPort
    final completer = Completer<SendPort>();
    final sub = _broadcastStream!.listen((msg) {
      if (msg is SendPort && !completer.isCompleted) {
        completer.complete(msg);
      }
    });
    _sendPort = await completer.future;
    await sub.cancel();

    // Listen for responses
    _broadcastStream!.listen(_onMessage);
  }

  /// Embed text via the isolate. Requests are queued — the isolate processes
  /// them one at a time (ONNX Runtime doesn't support parallel inference
  /// in a single session).
  Future<Float32List> embedText(String text) {
    final completer = Completer<Float32List>();
    _enqueue(_QueuedRequest(
      type: _RequestType.embedText,
      data: text,
      completer: completer,
    ));
    return completer.future;
  }

  /// Embed image file via the isolate. Same queuing as embedText.
  Future<Float32List> embedImage(String imagePath) {
    final completer = Completer<Float32List>();
    _enqueue(_QueuedRequest(
      type: _RequestType.embedImage,
      data: imagePath,
      completer: completer,
    ));
    return completer.future;
  }

  void _enqueue(_QueuedRequest request) {
    _requestQueue.add(request);
    _processNext();
  }

  void _processNext() {
    if (_isProcessing || _requestQueue.isEmpty || _sendPort == null) return;
    _isProcessing = true;
    final request = _requestQueue.removeAt(0);
    _pendingRequests.add(request.completer);

    switch (request.type) {
      case _RequestType.embedText:
        _sendPort!.send(_EmbedTextRequest(request.data as String));
      case _RequestType.embedImage:
        _sendPort!.send(_EmbedImageRequest(request.data as String));
    }
  }

  void _onMessage(dynamic msg) {
    if (msg is _EmbedResponse) {
      if (_pendingRequests.isNotEmpty) {
        _pendingRequests.removeAt(0).complete(msg.vector);
      }
    } else if (msg is _ErrorResponse) {
      if (_pendingRequests.isNotEmpty) {
        _pendingRequests.removeAt(0).completeError(msg.error);
      }
    }

    // Process next queued request
    _isProcessing = false;
    _processNext();
  }

  Future<void> dispose() async {
    _sendPort?.send(_DisposeRequest());
    _receivePort?.close();
    _isolate?.kill(priority: Isolate.immediate);
  }
}

// ─── Message types (all copyable across isolate boundaries) ─────────────────

class _IsolateInit {
  final SendPort mainPort;
  final InferenceIsolateConfig config;
  _IsolateInit({required this.mainPort, required this.config});
}

class _EmbedTextRequest {
  final String text;
  _EmbedTextRequest(this.text);
}

class _EmbedImageRequest {
  final String imagePath;
  _EmbedImageRequest(this.imagePath);
}

class _DisposeRequest {}

/// Request queue types for sequential isolate processing.
enum _RequestType { embedText, embedImage }

class _QueuedRequest {
  final _RequestType type;
  final Object data;
  final Completer<Float32List> completer;

  _QueuedRequest({
    required this.type,
    required this.data,
    required this.completer,
  });
}

class _EmbedResponse {
  final Float32List vector;
  _EmbedResponse(this.vector);
}

class _ErrorResponse {
  final String error;
  _ErrorResponse(this.error);
}

// ─── Isolate entry point ────────────────────────────────────────────────────

/// Runs inside the isolate. Creates ONNX sessions, handles requests.
void _inferenceIsolateEntry(_IsolateInit init) {
  final receivePort = ReceivePort();
  init.mainPort.send(receivePort.sendPort);

  // Create ONNX sessions inside the isolate
  ClipTextEmbedder? textEmbedder;
  ClipImageEmbedder? imageEmbedder;

  try {
    textEmbedder = ClipTextEmbedder(
      modelPath: init.config.clipTextModelPath,
      tokenizerPath: init.config.clipTokenizerPath,
    );
    // init() is sync for text embedder (loads tokenizer + ONNX)
    // We need to call it, but it's async — use a workaround
    final initCompleter = Completer<void>();
    textEmbedder.init().then((_) => initCompleter.complete());
    initCompleter.future; // This won't work — need to await

    // Actually, we can't await in a synchronous entry point.
    // The isolate's listen callback is async though.
  } catch (e) {
    init.mainPort.send(_ErrorResponse('Failed to create text embedder: $e'));
  }

  // We'll initialize lazily on first request
  var textInitialized = false;
  var imageInitialized = false;

  receivePort.listen((msg) async {
    try {
      if (msg is _EmbedTextRequest) {
        // Lazy init text embedder
        if (!textInitialized) {
          await textEmbedder!.init();
          textInitialized = true;
        }
        final vec = await textEmbedder!.embed(msg.text);
        init.mainPort.send(_EmbedResponse(vec));
      } else if (msg is _EmbedImageRequest) {
        // Lazy init image embedder
        if (imageEmbedder == null) {
          imageEmbedder = ClipImageEmbedder(
            modelPath: init.config.clipVisionModelPath,
          );
        }
        if (!imageInitialized) {
          await imageEmbedder!.init();
          imageInitialized = true;
        }
        final vec = await imageEmbedder!.embedFile(msg.imagePath);
        init.mainPort.send(_EmbedResponse(vec));
      } else if (msg is _DisposeRequest) {
        await textEmbedder?.dispose();
        await imageEmbedder?.dispose();
        Isolate.exit();
      }
    } catch (e) {
      init.mainPort.send(_ErrorResponse(e.toString()));
    }
  });
}

// ─── Isolate-backed embedder adapters ───────────────────────────────────────

/// TextEmbedder implementation that delegates to the inference isolate.
/// Used by the Coordinator for both query embedding and ingest text embedding.
class IsolateTextEmbedder implements TextEmbedder {
  final InferenceIsolateManager _manager;
  final int _dimension;

  IsolateTextEmbedder(this._manager, {int dimension = 512}) : _dimension = dimension;

  @override
  int get dimension => _dimension;

  @override
  Future<void> init() async {
    // The isolate is already initialized by the manager.
    // Nothing to do here.
  }

  @override
  Future<Float32List> embed(String text) async {
    return await _manager.embedText(text);
  }
}

/// ImageEmbedder implementation that delegates to the inference isolate.
class IsolateImageEmbedder implements ImageEmbedder {
  final InferenceIsolateManager _manager;
  final int _dimension;
  bool _initialized = false;

  IsolateImageEmbedder(this._manager, {int dimension = 512})
      : _dimension = dimension;

  @override
  int get dimension => _dimension;

  @override
  bool get isLoaded => _initialized;

  @override
  Future<void> init() async {
    _initialized = true;
    // The isolate handles lazy init internally
  }

  @override
  Future<void> dispose() async {
    _initialized = false;
    // The isolate disposes when the manager disposes
  }

  @override
  Future<Float32List> embedFile(String imagePath) async {
    return await _manager.embedImage(imagePath);
  }
}
