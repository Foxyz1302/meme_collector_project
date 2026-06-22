/// Concrete InferenceFactory implementation for the Windows app.
///
/// Creates the ONNX embedders that the Coordinator uses.
/// This is the bridge between the pure-Dart core package and the
/// Flutter-dependent ONNX code.

library;

import 'package:meme_collector_core/meme_collector_core.dart';
import 'package:path/path.dart' as p;

import 'inference_impl.dart';

/// Factory that creates concrete embedder implementations for the app.
///
/// All model paths are resolved relative to the assets/models/ directory.
/// On Windows desktop, assets live at `<workspace_root>/assets/models/`.
class AppInferenceFactory implements InferenceFactory {
  final String modelsDir;

  AppInferenceFactory({required this.modelsDir});

  String get _clipTextPath => p.join(modelsDir, 'clip_text_fp16.onnx');
  String get _clipVisionPath => p.join(modelsDir, 'clip_vision_fp16.onnx');
  String get _clipTokenizerPath => p.join(modelsDir, 'clip_tokenizer.json');

  @override
  TextEmbedder createQueryEmbedder() {
    // Using CLIP text tower for query embedding (model2vec removed due to
    // Native Assets incompatibility with Flutter 3.13 beta SDK).
    // ~70ms per query — well within the 150ms debounce budget.
    return ClipTextEmbedderAdapter(ClipTextEmbedder(
      modelPath: _clipTextPath,
      tokenizerPath: _clipTokenizerPath,
    ));
  }

  @override
  TextEmbedder? createClipTextEmbedder() {
    // Same CLIP text tower — used for cross-modal search (text → image).
    // The Coordinator calls this separately so it can be null for text-only
    // search. We return the same instance type.
    return ClipTextEmbedderAdapter(ClipTextEmbedder(
      modelPath: _clipTextPath,
      tokenizerPath: _clipTokenizerPath,
    ));
  }

  @override
  TextEmbedder createIngestTextEmbedder() {
    // Same as query embedder
    return ClipTextEmbedderAdapter(ClipTextEmbedder(
      modelPath: _clipTextPath,
      tokenizerPath: _clipTokenizerPath,
    ));
  }

  @override
  ImageEmbedder? createImageEmbedder() {
    return ClipImageEmbedder(modelPath: _clipVisionPath);
  }

  @override
  OcrEngine? createOcrEngine() {
    // TODO: implement PpocrEngine (det + rec pipeline)
    return null;
  }
}
