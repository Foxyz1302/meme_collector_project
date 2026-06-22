/// Isolate plumbing + Coordinator.
///
/// Two long-running isolates:
///   - Search isolate — owns the VectorIndex + KeywordIndex, handles search queries
///   - Ingest isolate — owns ONNX sessions + ffmpeg, processes ingest jobs
///
/// The main isolate owns metadata.json (single-writer pattern). Ingest
/// isolate sends progress events → main updates in-memory metadata +
/// schedules debounced save → main sends vector updates to search isolate.
///
/// The Coordinator is the public face of core. The app calls:
///   coordinator.search(query)
///   coordinator.addReaction(url, title, tags)
///   coordinator.deleteReaction(id)
///   coordinator.incrementUsage(id)
///
/// All isolate communication is via typed SendPort/ReceivePort messages.
library;

import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:path/path.dart' as p;

import 'inference.dart';
import 'ingest.dart';
import 'models.dart';
import 'search.dart';
import 'storage.dart';

// ─── Message protocol (typed, sent between isolates) ────────────────────────

/// Base class for all messages sent between isolates.
sealed class WorkerMessage {}

// Main → Search isolate
class SearchRequest extends WorkerMessage {
  final String query;
  final int topK;
  final Float32List? textQueryVec; // model2vec embedding (may be null if embedder fails)
  final Float32List? clipQueryVec; // CLIP text embedding (optional, for image search)

  SearchRequest({
    required this.query,
    required this.topK,
    this.textQueryVec,
    this.clipQueryVec,
  });
}

class HotlistRequest extends WorkerMessage {
  final int limit;
  HotlistRequest(this.limit);
}

class VectorIndexAddText extends WorkerMessage {
  final String reactionId;
  final Float32List vector;
  VectorIndexAddText(this.reactionId, this.vector);
}

class VectorIndexAddImage extends WorkerMessage {
  final String reactionId;
  final Float32List vector;
  VectorIndexAddImage(this.reactionId, this.vector);
}

class VectorIndexRemove extends WorkerMessage {
  final String reactionId;
  VectorIndexRemove(this.reactionId);
}

class VectorIndexRebuild extends WorkerMessage {
  final String storageRootPath;
  final String activeTextVersion;
  final String activeImageVersion;
  VectorIndexRebuild({
    required this.storageRootPath,
    required this.activeTextVersion,
    required this.activeImageVersion,
  });
}

// Search isolate → Main
class SearchResponse extends WorkerMessage {
  final List<SearchResult> results;
  SearchResponse(this.results);
}

class SearchError extends WorkerMessage {
  final String error;
  SearchError(this.error);
}

// Main → Ingest isolate
class IngestRequest extends WorkerMessage {
  final Reaction reaction;
  IngestRequest(this.reaction);
}

class IngestCancel extends WorkerMessage {
  final String reactionId;
  IngestCancel(this.reactionId);
}

class IngestShutdown extends WorkerMessage {}

// Ingest isolate → Main
class IngestProgressMsg extends WorkerMessage {
  final String reactionId;
  final ReactionStatus status;
  final double progress;
  IngestProgressMsg({
    required this.reactionId,
    required this.status,
    required this.progress,
  });
}

class IngestCompleteMsg extends WorkerMessage {
  final Reaction reaction;
  IngestCompleteMsg(this.reaction);
}

class IngestFailedMsg extends WorkerMessage {
  final String reactionId;
  final String error;
  IngestFailedMsg({required this.reactionId, required this.error});
}

// ─── Coordinator ────────────────────────────────────────────────────────────

/// The public API of meme_collector_core. Owns the metadata, spawns the
/// search + ingest isolates, and exposes high-level methods the UI calls.
///
/// Lifecycle:
///   final coordinator = Coordinator(config: ...);
///   await coordinator.init();  // spawns isolates, loads metadata + vectors
///   // ... use coordinator ...
///   await coordinator.dispose();  // shuts down isolates, flushes metadata
class Coordinator {
  final CoordinatorConfig config;

  // Storage
  late final Storage _storage;
  Metadata _metadata = Metadata.empty();

  // Inference (resident in main isolate for query embedding)
  late final TextEmbedder _queryEmbedder; // model2vec for fast text search
  late final TextEmbedder? _clipTextEmbedder; // CLIP text tower (optional, for cross-modal)

  // Ingest components (run in ingest isolate)
  late final TextEmbedder _ingestTextEmbedder;
  late final ImageEmbedder? _ingestImageEmbedder;
  late final OcrEngine? _ocrEngine;
  late final FfmpegWrapper? _ffmpeg;

  // Isolates
  SearchIsolate? _searchIso;
  IngestIsolate? _ingestIso;

  // Save debouncing
  Timer? _saveDebounce;
  bool _disposed = false;

  // Stream controllers for the UI to listen
  final _progressController = StreamController<IngestProgressMsg>.broadcast();
  final _completeController = StreamController<IngestCompleteMsg>.broadcast();
  final _failedController = StreamController<IngestFailedMsg>.broadcast();

  Coordinator({required this.config});

  // ─── Streams the UI can listen to ──────────────────────────────────────

  /// Emits when a reaction's ingest progress changes.
  Stream<IngestProgressMsg> get progressStream => _progressController.stream;

  /// Emits when a reaction finishes ingesting.
  Stream<IngestCompleteMsg> get completeStream => _completeController.stream;

  /// Emits when a reaction fails to ingest.
  Stream<IngestFailedMsg> get failedStream => _failedController.stream;

  // ─── Initialization ───────────────────────────────────────────────────

  /// Initialize the coordinator. Spawns isolates, loads metadata + vectors.
  ///
  /// [inferenceFactory] provides the concrete embedder implementations
  /// (the app passes in ONNX-based ones; the future server would pass
  /// different ones).
  Future<void> init(InferenceFactory inferenceFactory) async {
    _storage = Storage(config.storagePath);
    await _storage.ensureDirectoriesExist();

    // Load metadata
    _metadata = await _storage.loadMetadata();

    // Create embedders
    _queryEmbedder = inferenceFactory.createQueryEmbedder();
    await _queryEmbedder.init();

    _clipTextEmbedder = inferenceFactory.createClipTextEmbedder();
    if (_clipTextEmbedder != null) {
      await _clipTextEmbedder.init();
    }

    _ingestTextEmbedder = inferenceFactory.createIngestTextEmbedder();
    await _ingestTextEmbedder.init();

    _ingestImageEmbedder = inferenceFactory.createImageEmbedder();
    _ocrEngine = inferenceFactory.createOcrEngine();
    _ffmpeg = await FfmpegWrapper.detect();

    // Spawn search isolate
    _searchIso = SearchIsolate();
    await _searchIso!.spawn();
    _searchIso!.responses.listen(_onSearchResponse);

    // Tell search isolate to load the vector index
    _searchIso!.send(VectorIndexRebuild(
      storageRootPath: _storage.rootPath,
      activeTextVersion: config.activeTextModelVersion,
      activeImageVersion: config.activeImageModelVersion,
    ));

    // Spawn ingest isolate
    _ingestIso = IngestIsolate();
    await _ingestIso!.spawn(
      storageRootPath: _storage.rootPath,
      textEmbedder: _ingestTextEmbedder,
      imageEmbedder: _ingestImageEmbedder,
      ocrEngine: _ocrEngine,
      ffmpeg: _ffmpeg,
      ingestConfig: config.ingestConfig,
    );
    _ingestIso!.events.listen(_onIngestEvent);

    // Re-queue any incomplete reactions (crash recovery)
    _requeueIncomplete();
  }

  /// Re-queue reactions that were mid-ingest when the app last closed.
  void _requeueIncomplete() {
    for (final r in _metadata.reactions) {
      if (r.status != ReactionStatus.ready &&
          r.status != ReactionStatus.failed) {
        // Reset to queued and re-ingest
        final reset = r.copyWith(
          status: ReactionStatus.queued,
          progress: 0.0,
          errorMessage: null,
        );
        _updateReaction(reset);
        _ingestIso?.send(IngestRequest(reset));
      }
    }
  }

  // ─── Public API (called by the UI) ────────────────────────────────────

  /// Search for reactions matching [query].
  ///
  /// Empty query returns the hotlist (most-used reactions).
  Future<List<SearchResult>> search(String query, {int topK = 50}) async {
    if (query.isEmpty) {
      return hotlist(limit: topK);
    }

    // Embed query in main isolate (fast, doesn't need ingest isolate)
    Float32List? textVec;
    Float32List? clipVec;
    try {
      textVec = await _queryEmbedder.embed(query);
    } catch (_) {
      // model2vec failed — fall back to keyword-only search
    }

    if (_clipTextEmbedder != null) {
      try {
        clipVec = await _clipTextEmbedder.embed(query);
      } catch (_) {
        // CLIP failed — text-only search
      }
    }

    // Send to search isolate
    final completer = Completer<List<SearchResult>>();
    _pendingSearches.add(completer);

    _searchIso?.send(SearchRequest(
      query: query,
      topK: topK,
      textQueryVec: textVec,
      clipQueryVec: clipVec,
    ));

    return completer.future;
  }

  /// Get the hotlist (most-used reactions).
  Future<List<SearchResult>> hotlist({int limit = 60}) async {
    final completer = Completer<List<SearchResult>>();
    _pendingHotlists.add(completer);
    _searchIso?.send(HotlistRequest(limit));
    return completer.future;
  }

  /// Add a new reaction from a URL.
  ///
  /// Returns the created Reaction, or null if the URL is a duplicate.
  Future<Reaction?> addReaction({
    required String url,
    String? title,
    List<String>? tags,
  }) async {
    final factory = ReactionFactory();
    final reaction = await factory.create(
      url: url,
      title: title,
      tags: tags,
      existingReactions: _metadata.reactions,
    );

    if (reaction == null) return null; // duplicate

    // Add to metadata + save
    _metadata = _metadata.copyWith(reactions: [..._metadata.reactions, reaction]);
    _scheduleSave();

    // Compute initial text embedding from title + tags (before download)
    // so the reaction is immediately searchable
    if (reaction.embeddableText.isNotEmpty) {
      try {
        final textVec = await _queryEmbedder.embed(reaction.embeddableText);
        final vecPath = _storage.textEmbeddingPath(
            reaction.id, 'model2vec', config.activeTextModelVersion);
        await writeVectorFile(vecPath, textVec);

        // Update reaction with embedding path
        final updated = reaction.copyWith(
          textEmbeddingPath: vecPath
              .substring(_storage.rootPath.length + 1)
              .replaceAll('\\', '/'),
          textModelVersion: config.activeTextModelVersion,
        );
        _updateReaction(updated);

        // Tell search isolate about the new vector
        _searchIso?.send(VectorIndexAddText(reaction.id, textVec));
      } catch (_) {
        // Embedding failed — reaction still queued for ingest
      }
    }

    // Queue for ingest (download + thumbnails + image embedding + OCR)
    _ingestIso?.send(IngestRequest(reaction));

    return reaction;
  }

  /// Delete a reaction. Removes from metadata + deletes derivative files.
  Future<void> deleteReaction(String id) async {
    final reaction = _metadata.byId(id);
    if (reaction == null) return;

    // Remove from metadata
    _metadata = _metadata.copyWith(
        reactions: _metadata.reactions.where((r) => r.id != id).toList());
    _scheduleSave();

    // Tell search isolate to remove from index
    _searchIso?.send(VectorIndexRemove(id));

    // Delete derivative files on disk
    await _storage.deleteReactionFiles(reaction);
  }

  /// Increment usage count + update lastUsedAt (called when user clicks a reaction).
  Future<void> incrementUsage(String id) async {
    final reaction = _metadata.byId(id);
    if (reaction == null) return;

    final updated = reaction.copyWith(
      usageCount: reaction.usageCount + 1,
      lastUsedAt: DateTime.now(),
    );
    _updateReaction(updated);
    _scheduleSave();
  }

  /// Update a reaction's metadata (title, tags, notes).
  Future<void> updateMetadata(
    String id, {
    String? title,
    List<String>? tags,
    String? notes,
  }) async {
    final reaction = _metadata.byId(id);
    if (reaction == null) return;

    var updated = reaction;
    if (title != null) updated = updated.copyWith(title: title);
    if (tags != null) updated = updated.copyWith(tags: tags);
    if (notes != null) updated = updated.copyWith(notes: notes);
    _updateReaction(updated);
    _scheduleSave();

    // If text metadata changed, re-embed + update search index
    if (title != null || tags != null) {
      try {
        final textVec = await _queryEmbedder.embed(updated.embeddableText);
        final vecPath = _storage.textEmbeddingPath(
            id, 'model2vec', config.activeTextModelVersion);
        await writeVectorFile(vecPath, textVec);
        _searchIso?.send(VectorIndexAddText(id, textVec));
      } catch (_) {
        // re-embed failed — keep old embedding
      }
    }
  }

  /// Pin a reaction locally (force download if not already downloaded).
  Future<void> pinReaction(String id) async {
    final reaction = _metadata.byId(id);
    if (reaction == null) return;

    if (reaction.pinned) return; // already pinned

    final updated = reaction.copyWith(pinned: true);
    _updateReaction(updated);
    _scheduleSave();

    // If not yet downloaded, queue for ingest
    if (reaction.localFile == null) {
      _ingestIso?.send(IngestRequest(updated));
    }
  }

  /// Get a reaction by ID (for UI display).
  Reaction? getReaction(String id) => _metadata.byId(id);

  /// Get all reactions (for UI grid display).
  List<Reaction> get allReactions => List.unmodifiable(_metadata.reactions);

  /// The storage root path. Used by the UI to resolve relative file paths
  /// (thumbnails, local files) to absolute paths for Image.file().
  String get storagePath => _storage.rootPath;

  /// Resolve a relative path (stored in Reaction) to an absolute path.
  String resolvePath(String relative) {
    // Normalize separators — relative paths use / on all platforms
    final normalized = relative.replaceAll('/', p.separator);
    return p.join(_storage.rootPath, normalized);
  }

  /// Force-write any pending debounced save. Call on app exit.
  Future<void> flushNow() async {
    _saveDebounce?.cancel();
    await _doSave();
  }

  /// Shut down isolates + flush metadata. Call on app exit.
  Future<void> dispose() async {
    if (_disposed) return;
    _disposed = true;

    _saveDebounce?.cancel();
    await _doSave();

    _ingestIso?.send(IngestShutdown());
    await _ingestIso?.dispose();
    await _searchIso?.dispose();

    await _progressController.close();
    await _completeController.close();
    await _failedController.close();
  }

  // ─── Internal ─────────────────────────────────────────────────────────

  final _pendingSearches = <Completer<List<SearchResult>>>[];
  final _pendingHotlists = <Completer<List<SearchResult>>>[];

  void _onSearchResponse(WorkerMessage msg) {
    if (msg is SearchResponse) {
      if (_pendingSearches.isNotEmpty) {
        _pendingSearches.removeAt(0).complete(msg.results);
      }
    } else if (msg is SearchError) {
      if (_pendingSearches.isNotEmpty) {
        _pendingSearches.removeAt(0).complete([]);
      }
    }
  }

  void _onIngestEvent(WorkerMessage msg) {
    switch (msg) {
      case IngestProgressMsg(:final reactionId, :final status, :final progress):
        final reaction = _metadata.byId(reactionId);
        if (reaction != null) {
          _updateReaction(reaction.copyWith(
            status: status,
            progress: progress,
          ));
          _scheduleSave();
        }
        _progressController.add(msg);
        break;

      case IngestCompleteMsg(:final reaction):
        _updateReaction(reaction);
        _scheduleSave();

        // Tell search isolate about new vectors
        if (reaction.textEmbeddingPath != null && reaction.textModelVersion != null) {
          _loadAndSendVector(reaction.id, reaction.textEmbeddingPath!, isImage: false);
        }
        if (reaction.imageEmbeddingPath != null) {
          _loadAndSendVector(reaction.id, reaction.imageEmbeddingPath!, isImage: true);
        }

        _completeController.add(msg);
        break;

      case IngestFailedMsg(:final reactionId, :final error):
        final reaction = _metadata.byId(reactionId);
        if (reaction != null) {
          _updateReaction(reaction.copyWith(
            status: ReactionStatus.failed,
            errorMessage: error,
          ));
          _scheduleSave();
        }
        _failedController.add(msg);
        break;

      default:
        // ignore unexpected messages
        break;
    }
  }

  /// Load a vector from disk and send it to the search isolate.
  Future<void> _loadAndSendVector(
    String reactionId,
    String relativePath, {
    required bool isImage,
  }) async {
    try {
      final absPath =
          '${_storage.rootPath}/${relativePath.replaceAll('/', p.separator)}';
      final vec = readVectorFile(absPath);
      if (isImage) {
        _searchIso?.send(VectorIndexAddImage(reactionId, vec));
      } else {
        _searchIso?.send(VectorIndexAddText(reactionId, vec));
      }
    } catch (_) {
      // file missing — skip
    }
  }

  /// Update a reaction in the in-memory metadata (immutable replace).
  void _updateReaction(Reaction updated) {
    final newReactions = _metadata.reactions
        .map((r) => r.id == updated.id ? updated : r)
        .toList();
    _metadata = _metadata.copyWith(reactions: newReactions);
  }

  void _scheduleSave() {
    _saveDebounce?.cancel();
    _saveDebounce = Timer(const Duration(milliseconds: 500), _doSave);
  }

  Future<void> _doSave() async {
    if (_disposed) return;
    await _storage.saveNow(_metadata);
  }
}

// ─── Coordinator config ─────────────────────────────────────────────────────

/// Configuration for the Coordinator.
class CoordinatorConfig {
  /// Path to the storage root (where metadata.json + derivative files live).
  final String storagePath;

  /// Whether animated previews are enabled.
  final bool animatedPreviewsEnabled;

  /// Whether OCR is enabled.
  final bool ocrEnabled;

  /// Active model versions (must match what the embedders produce).
  final String activeTextModelVersion;
  final String activeImageModelVersion;
  final String activeOcrModelVersion;

  const CoordinatorConfig({
    required this.storagePath,
    this.animatedPreviewsEnabled = false,
    this.ocrEnabled = false,
    this.activeTextModelVersion = 'potion-base-32M',
    this.activeImageModelVersion = 'clip-vit-b32-fp16-v1',
    this.activeOcrModelVersion = 'pp-ocr-v5',
  });

  /// Convert to IngestConfig for the ingest pipeline.
  IngestConfig get ingestConfig => IngestConfig(
        animatedPreviewsEnabled: animatedPreviewsEnabled,
        ocrEnabled: ocrEnabled,
        textModelVersion: activeTextModelVersion,
        imageModelVersion: activeImageModelVersion,
        ocrModelVersion: activeOcrModelVersion,
      );
}

// ─── Inference Factory ──────────────────────────────────────────────────────

/// Factory that creates concrete embedder implementations.
///
/// The app provides an implementation that creates ONNX-based embedders.
/// The future server would provide a different implementation.
///
/// This abstraction lets the core Coordinator stay pure-Dart while the
/// app injects the Flutter-dependent ONNX code.
abstract class InferenceFactory {
  /// Fast text embedder for query embedding (model2vec).
  /// Used in the main isolate for instant query embedding.
  TextEmbedder createQueryEmbedder();

  /// CLIP text embedder for cross-modal search (optional).
  /// Returns null if cross-modal search is disabled.
  TextEmbedder? createClipTextEmbedder();

  /// Text embedder for the ingest pipeline (model2vec, used for re-embedding).
  /// Can be the same instance as createQueryEmbedder.
  TextEmbedder createIngestTextEmbedder();

  /// Image embedder for the ingest pipeline (CLIP vision tower).
  /// Returns null if image embeddings are disabled.
  ImageEmbedder? createImageEmbedder();

  /// OCR engine for the ingest pipeline (PP-OCRv5).
  /// Returns null if OCR is disabled.
  OcrEngine? createOcrEngine();
}

// ─── Search Isolate ─────────────────────────────────────────────────────────

/// Long-running isolate that owns the VectorIndex + KeywordIndex.
///
/// Receives search requests + vector index updates, sends back search results.
class SearchIsolate {
  SendPort? _sendPort;
  ReceivePort? _receivePort;
  Isolate? _isolate;
  Stream<WorkerMessage>? _responseStream;

  /// Stream of responses from the search isolate.
  /// Uses broadcast stream so multiple listeners can subscribe.
  Stream<WorkerMessage> get responses =>
      _responseStream ?? const Stream.empty();

  Future<void> spawn() async {
    _receivePort = ReceivePort();
    // Convert to broadcast stream so both the handshake + the public
    // responses stream can listen without "already listened to" errors.
    final broadcast = _receivePort!.asBroadcastStream();
    _responseStream = broadcast.whereType<WorkerMessage>();

    _isolate = await Isolate.spawn(_searchIsolateEntry, _receivePort!.sendPort);

    // Wait for the isolate to send us its SendPort
    final completer = Completer<SendPort>();
    final sub = broadcast.listen((msg) {
      if (msg is SendPort && !completer.isCompleted) {
        completer.complete(msg);
      }
    });
    _sendPort = await completer.future;
    await sub.cancel();
  }

  void send(WorkerMessage msg) {
    _sendPort?.send(msg);
  }

  Future<void> dispose() async {
    _receivePort?.close();
    _isolate?.kill(priority: Isolate.immediate);
  }
}

/// Entry point for the search isolate.
void _searchIsolateEntry(SendPort mainPort) {
  final receivePort = ReceivePort();
  mainPort.send(receivePort.sendPort);

  final vectorIndex = VectorIndex();
  final keywordIndex = KeywordIndex();
  // reactionsById is rebuilt whenever we get a VectorIndexRebuild or hotlist request
  var reactionsById = <String, Reaction>{};

  final service = SearchService(
    vectorIndex: vectorIndex,
    keywordIndex: keywordIndex,
  );

  receivePort.listen((msg) {
    if (msg is! WorkerMessage) return;

    switch (msg) {
      case VectorIndexRebuild(:final storageRootPath, :final activeTextVersion, :final activeImageVersion):
        // Load vectors from disk (async)
        () async {
          final storage = Storage(storageRootPath);
          await vectorIndex.load(storage, activeTextVersion, activeImageVersion);

          // Rebuild keyword index from metadata
          final metadata = await storage.loadMetadata();
          reactionsById = {for (final r in metadata.reactions) r.id: r};
          keywordIndex.rebuild(metadata.reactions);
        }();

      case SearchRequest(:final query, :final topK, :final textQueryVec, :final clipQueryVec):
        if (textQueryVec == null) {
          // Text embedding failed — keyword-only search
          final keywordHits = keywordIndex.search(query, topK: topK);
          final results = keywordHits
              .map((h) => SearchResult(
                    reactionId: h.$1,
                    score: h.$2,
                    rrfScore: h.$2,
                    usageBoost: 0,
                  ))
              .toList();
          mainPort.send(SearchResponse(results));
        } else {
          final results = service.search(
            query: query,
            textQueryVec: textQueryVec,
            clipQueryVec: clipQueryVec,
            reactionsById: reactionsById,
            topK: topK,
          );
          mainPort.send(SearchResponse(results));
        }

      case HotlistRequest(:final limit):
        final results = service.hotlist(
          reactionsById: reactionsById,
          limit: limit,
        );
        mainPort.send(SearchResponse(results));

      case VectorIndexAddText(:final reactionId, :final vector):
        vectorIndex.addTextVector(reactionId, vector);

      case VectorIndexAddImage(:final reactionId, :final vector):
        vectorIndex.addImageVector(reactionId, vector);

      case VectorIndexRemove(:final reactionId):
        vectorIndex.remove(reactionId);
        keywordIndex.remove(reactionId);
        reactionsById.remove(reactionId);

      default:
        // ignore unexpected messages
        break;
    }
  });
}

// ─── Ingest Isolate ─────────────────────────────────────────────────────────

/// Long-running isolate that owns ONNX sessions + ffmpeg.
///
/// Receives ingest requests, runs the pipeline, sends back progress + completion.
class IngestIsolate {
  SendPort? _sendPort;
  ReceivePort? _receivePort;
  Isolate? _isolate;
  Stream<WorkerMessage>? _eventStream;

  /// Stream of events from the ingest isolate.
  /// Uses broadcast stream so multiple listeners can subscribe.
  Stream<WorkerMessage> get events =>
      _eventStream ?? const Stream.empty();

  Future<void> spawn({
    required String storageRootPath,
    required TextEmbedder textEmbedder,
    ImageEmbedder? imageEmbedder,
    OcrEngine? ocrEngine,
    FfmpegWrapper? ffmpeg,
    required IngestConfig ingestConfig,
  }) async {
    _receivePort = ReceivePort();
    // Convert to broadcast stream so both the handshake + the public
    // events stream can listen without "already listened to" errors.
    final broadcast = _receivePort!.asBroadcastStream();
    _eventStream = broadcast.whereType<WorkerMessage>();

    final init = _IngestIsolateInit(
      mainPort: _receivePort!.sendPort,
      storageRootPath: storageRootPath,
      textEmbedder: textEmbedder,
      imageEmbedder: imageEmbedder,
      ocrEngine: ocrEngine,
      ffmpeg: ffmpeg,
      ingestConfig: ingestConfig,
    );

    _isolate = await Isolate.spawn(_ingestIsolateEntry, init);

    // Wait for the isolate to send us its SendPort
    final completer = Completer<SendPort>();
    final sub = broadcast.listen((msg) {
      if (msg is SendPort && !completer.isCompleted) {
        completer.complete(msg);
      }
    });
    _sendPort = await completer.future;
    await sub.cancel();
  }

  void send(WorkerMessage msg) {
    _sendPort?.send(msg);
  }

  Future<void> dispose() async {
    _receivePort?.close();
    _isolate?.kill(priority: Isolate.immediate);
  }
}

/// Init payload for the ingest isolate.
class _IngestIsolateInit {
  final SendPort mainPort;
  final String storageRootPath;
  final TextEmbedder textEmbedder;
  final ImageEmbedder? imageEmbedder;
  final OcrEngine? ocrEngine;
  final FfmpegWrapper? ffmpeg;
  final IngestConfig ingestConfig;

  _IngestIsolateInit({
    required this.mainPort,
    required this.storageRootPath,
    required this.textEmbedder,
    required this.imageEmbedder,
    required this.ocrEngine,
    required this.ffmpeg,
    required this.ingestConfig,
  });
}

/// Entry point for the ingest isolate.
void _ingestIsolateEntry(_IngestIsolateInit init) {
  final receivePort = ReceivePort();
  init.mainPort.send(receivePort.sendPort);

  final storage = Storage(init.storageRootPath);
  final pipeline = IngestPipeline(
    storage: storage,
    textEmbedder: init.textEmbedder,
    imageEmbedder: init.imageEmbedder,
    ocrEngine: init.ocrEngine,
    ffmpeg: init.ffmpeg,
    dio: Dio(),
    config: init.ingestConfig,
  );

  receivePort.listen((msg) {
    if (msg is! WorkerMessage) return;

    switch (msg) {
      case IngestRequest(:final reaction):
        // Run the pipeline, forward events to main isolate
        () async {
          await for (final event in pipeline.process(reaction)) {
            switch (event) {
              case IngestProgressEvent(:final reactionId, :final status, :final progress):
                init.mainPort.send(IngestProgressMsg(
                  reactionId: reactionId,
                  status: status,
                  progress: progress,
                ));
              case IngestCompleteEvent(:final reaction):
                init.mainPort.send(IngestCompleteMsg(reaction));
              case IngestFailedEvent(:final reactionId, :final error):
                init.mainPort.send(IngestFailedMsg(
                  reactionId: reactionId,
                  error: error,
                ));
            }
          }
        }();

      case IngestCancel():
        // TODO: implement cancellation
        break;

      case IngestShutdown():
        Isolate.exit();

      default:
        break;
    }
  });
}
