/// Abstract inference interfaces.
///
/// Core defines these as abstract so the ingest pipeline and search service
/// can reference them without depending on Flutter-only packages (ONNX Runtime
/// is a Flutter plugin). The app provides concrete implementations using
/// flutter_onnxruntime + model2vec.
///
/// When you build the future server, it provides its own implementations
/// (could be the same ONNX code, or a Python subprocess, or a remote API).
library;

import 'dart:math' as math;
import 'dart:typed_data';

/// Embeds text into a fixed-dim Float32 vector (L2-normalized).
///
/// Used for:
///   - Query embedding at search time (fast, frequent)
///   - Reaction text embedding at ingest time (title + tags + OCR text)
abstract class TextEmbedder {
  /// Dim of vectors produced (e.g. 512 for model2vec potion-base-32M).
  int get dimension;

  /// Embed a single string. Returns L2-normalized Float32List of length [dimension].
  Future<Float32List> embed(String text);

  /// Initialize the model. Called once at startup.
  Future<void> init();
}

/// Embeds images into the same vector space as [TextEmbedder] (cross-modal).
///
/// Used for:
///   - Image embedding at ingest time (background, infrequent)
///   - "Search by example image" queries (optional feature)
abstract class ImageEmbedder {
  /// Dim of vectors produced. Must match [TextEmbedder.dimension] for cross-modal search.
  int get dimension;

  /// Embed an image file. Returns L2-normalized Float32List.
  Future<Float32List> embedFile(String imagePath);

  /// Initialize the model. Called lazily on first use.
  Future<void> init();

  /// Release the model from memory. Called after idle period.
  Future<void> dispose();

  /// True if the model is currently loaded in memory.
  bool get isLoaded;
}

/// OCR engine — extracts text from images.
///
/// Used for:
///   - Extracting overlaid meme text at ingest time (optional, lazy)
abstract class OcrEngine {
  /// Run OCR on an image file. Returns extracted text, or null if no text found.
  Future<String?> ocrFile(String imagePath);

  /// Initialize the OCR models. Called lazily on first use.
  Future<void> init();

  /// Release the models from memory. Called after idle period.
  Future<void> dispose();

  /// True if the models are currently loaded in memory.
  bool get isLoaded;
}

/// L2-normalize a vector in place. All embedders should return normalized vectors
/// so cosine similarity = dot product (faster, no sqrt needed at query time).
void l2Normalize(Float32List v) {
  var sumSq = 0.0;
  for (var i = 0; i < v.length; i++) {
    sumSq += v[i] * v[i];
  }
  if (sumSq == 0.0) return;
  final invLen = 1.0 / math.sqrt(sumSq);
  for (var i = 0; i < v.length; i++) {
    v[i] = (v[i] * invLen).toDouble();
  }
}

/// Cosine similarity between two vectors.
///
/// If both vectors are pre-normalized (which all embedders in this package
/// guarantee), this is just a dot product. The general formula is used here
/// so it's correct even for un-normalized inputs.
double cosineSimilarity(Float32List a, Float32List b) {
  if (a.length != b.length) {
    throw ArgumentError(
        'Vectors must have same length (${a.length} vs ${b.length})');
  }
  var dot = 0.0;
  var normA = 0.0;
  var normB = 0.0;
  for (var i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA == 0.0 || normB == 0.0) return 0.0;
  return dot / math.sqrt(normA * normB);
}
