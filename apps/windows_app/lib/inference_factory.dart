/// Concrete InferenceFactory implementation for the Windows app.
///
/// Creates isolate-backed embedders that delegate ONNX inference to a
/// separate isolate, keeping the UI thread responsive.

library;

import 'package:meme_collector_core/meme_collector_core.dart';
import 'package:path/path.dart' as p;

import 'inference_isolate.dart';

/// Factory that creates isolate-backed embedder implementations.
///
/// All ONNX inference runs in a separate isolate. The main isolate sends
/// text/image paths, receives Float32List embeddings back. This keeps the
/// UI thread responsive during both search queries and ingest.
class AppInferenceFactory implements InferenceFactory {
  final String modelsDir;
  final InferenceIsolateManager _isolateManager;

  AppInferenceFactory({
    required this.modelsDir,
    required InferenceIsolateManager isolateManager,
  }) : _isolateManager = isolateManager;

  @override
  TextEmbedder createQueryEmbedder() {
    return IsolateTextEmbedder(_isolateManager);
  }

  @override
  TextEmbedder? createClipTextEmbedder() {
    return IsolateTextEmbedder(_isolateManager);
  }

  @override
  TextEmbedder createIngestTextEmbedder() {
    return IsolateTextEmbedder(_isolateManager);
  }

  @override
  ImageEmbedder? createImageEmbedder() {
    return IsolateImageEmbedder(_isolateManager);
  }

  @override
  OcrEngine? createOcrEngine() {
    return null; // TODO: implement PpocrEngine
  }
}
