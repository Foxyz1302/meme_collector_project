/// Concrete ONNX + model2vec implementations of the abstract inference
/// interfaces defined in meme_collector_core/inference.dart.
///
/// This file lives in the app (not core) because it depends on Flutter-only
/// packages (onnxruntime_v2, model2vec, image). Core stays pure-Dart for
/// the future server migration.
///
/// Four implementations:
///   - ClipTextEmbedder   — CLIP text tower for cross-modal query embedding
///   - ClipImageEmbedder   — CLIP vision tower for image embeddings
///   - Model2VecEmbedder   — model2vec for pure-text embedding (fast)
///   - PpocrEngine         — PP-OCRv5 det + rec pipeline (TODO)

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:meme_collector_core/meme_collector_core.dart';
import 'package:model2vec/model2vec.dart';
import 'package:onnxruntime_v2/onnxruntime_v2.dart';
import 'package:path/path.dart' as p;

// ─── CLIP BPE Tokenizer ─────────────────────────────────────────────────────

/// Byte-level BPE tokenizer for CLIP.
///
/// Loads from HuggingFace tokenizer.json format. CLIP uses:
///   - Lowercase text
///   - Regex pre-tokenization (words + punctuation + whitespace)
///   - Byte-level BPE (bytes → unicode chars → BPE merges)
///   - BOS (49406) + EOS (49407) wrapping, pad to 77 tokens with EOS
class ClipTokenizer {
  static const int bosTokenId = 49406;
  static const int eosTokenId = 49407;
  static const int contextLength = 77;

  final Map<String, int> _vocab;
  final List<(String, String)> _merges; // (a, b) pairs in priority order
  final Map<(String, String), int> _mergeRanks;

  ClipTokenizer._(this._vocab, this._merges, this._mergeRanks);

  /// Load tokenizer from a HuggingFace tokenizer.json file.
  factory ClipTokenizer.fromFile(String path) {
    final json = jsonDecode(File(path).readAsStringSync()) as Map<String, dynamic>;
    return ClipTokenizer.fromJson(json);
  }

  factory ClipTokenizer.fromJson(Map<String, dynamic> json) {
    final model = json['model'] as Map<String, dynamic>;
    final vocabRaw = model['vocab'] as Map<String, dynamic>;
    final vocab = vocabRaw.map((k, v) => MapEntry(k, v as int));

    final mergesRaw = model['merges'] as List<dynamic>;
    final merges = <(String, String)>[];
    final mergeRanks = <(String, String), int>{};
    for (var i = 0; i < mergesRaw.length; i++) {
      final parts = (mergesRaw[i] as String).split(' ');
      if (parts.length == 2) {
        final pair = (parts[0], parts[1]);
        merges.add(pair);
        mergeRanks[pair] = i;
      }
    }

    return ClipTokenizer._(vocab, merges, mergeRanks);
  }

  /// Encode a string to CLIP token IDs (length = contextLength = 77).
  ///
  /// Pattern: [BOS] tokens... [EOS] [PAD]... [PAD]
  /// Where PAD = EOS (49407) in CLIP.
  List<int> encode(String text) {
    final lowercased = text.toLowerCase();
    final preTokens = _preTokenize(lowercased);

    final tokenIds = <int>[];
    for (final preToken in preTokens) {
      final bytes = preToken.codeUnits;
      // Byte-level: convert each byte to its unicode char
      final unicodeChars = bytes.map(_byteToUnicode).join('');
      final bpeTokens = _bpe(unicodeChars);
      for (final token in bpeTokens) {
        final id = _vocab[token];
        if (id != null) {
          tokenIds.add(id);
        }
      }
    }

    // Truncate to contextLength - 2 (leave room for BOS + EOS)
    final maxBodyLen = contextLength - 2;
    if (tokenIds.length > maxBodyLen) {
      tokenIds.removeRange(maxBodyLen, tokenIds.length);
    }

    // Build final: [BOS] body [EOS] [PAD]... [PAD]
    final result = List<int>.filled(contextLength, eosTokenId);
    result[0] = bosTokenId;
    for (var i = 0; i < tokenIds.length; i++) {
      result[i + 1] = tokenIds[i];
    }
    // Position tokenIds.length + 1 is EOS (already set by the fill)
    return result;
  }

  /// CLIP pre-tokenization regex. Splits on:
  ///   - Whitespace
  ///   - Punctuation
  ///   - Contractions (apostrophes)
  ///   - Sequences of letters/digits
  static final _preTokenRegex = RegExp(
    r"""'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]+|[^\s\p{L}\p{N}]+""",
    unicode: true,
  );

  List<String> _preTokenize(String text) {
    return _preTokenRegex
        .allMatches(text)
        .map((m) => m.group(0)!)
        .toList();
  }

  /// Byte-to-unicode mapping for byte-level BPE.
  /// Maps each byte (0-255) to a unicode character.
  /// Bytes 33-126, 161-172, 174-255 map to themselves.
  /// Other bytes map to characters 256+.
  static final Map<int, String> _byteToUnicodeMap = _buildByteToUnicodeMap();

  static Map<int, String> _buildByteToUnicodeMap() {
    final map = <int, String>{};
    final bs = <int>[];
    for (var b = 0; b < 256; b++) {
      if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
        // maps to itself
      } else {
        bs.add(b);
      }
    }
    var c = 256;
    for (final b in bs) {
      map[b] = String.fromCharCode(c);
      c++;
    }
    for (var b = 0; b < 256; b++) {
      if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
        map[b] = String.fromCharCode(b);
      }
    }
    return map;
  }

  String _byteToUnicode(int byte) => _byteToUnicodeMap[byte] ?? '?';

  /// Apply BPE merges to a string of unicode chars.
  List<String> _bpe(String token) {
    if (token.isEmpty) return [];

    var word = token.split('');
    var pairs = _getPairs(word);

    while (true) {
      final minPair = _findMinPair(pairs);
      if (minPair == null) break;

      final newWord = <String>[];
      var i = 0;
      while (i < word.length) {
        var j = -1;
        for (var k = i; k < word.length - 1; k++) {
          if (word[k] == minPair.$1 && word[k + 1] == minPair.$2) {
            j = k;
            break;
          }
        }
        if (j == -1) {
          newWord.addAll(word.sublist(i));
          break;
        }
        newWord.addAll(word.sublist(i, j));
        newWord.add(minPair.$1 + minPair.$2);
        i = j + 2;
      }
      word = newWord;
      if (word.length == 1) break;
      pairs = _getPairs(word);
    });

    return word;
  }

  List<(String, String)> _getPairs(List<String> word) {
    final pairs = <(String, String)>[];
    for (var i = 0; i < word.length - 1; i++) {
      pairs.add((word[i], word[i + 1]));
    }
    return pairs;
  }

  (String, String)? _findMinPair(List<(String, String)> pairs) {
    if (pairs.isEmpty) return null;
    (String, String)? minPair;
    int? minRank;
    for (final pair in pairs) {
      final rank = _mergeRanks[pair];
      if (rank != null && (minRank == null || rank < minRank)) {
        minRank = rank;
        minPair = pair;
      }
    }
    return minPair;
  }
}

// ─── CLIP Text Embedder ─────────────────────────────────────────────────────

/// CLIP text tower — embeds text into the 512-dim CLIP vector space.
///
/// Used for cross-modal queries: "describe the reaction" → find by image
/// embedding similarity. NOT used for pure text search (that's model2vec).
class ClipTextEmbedder implements TextEmbedder {
  final String _modelPath;
  final String _tokenizerPath;
  final int _dimension;

  OrtSession? _session;
  ClipTokenizer? _tokenizer;
  bool _initialized = false;

  ClipTextEmbedder({
    required String modelPath,
    required String tokenizerPath,
    int dimension = 512,
  })  : _modelPath = modelPath,
        _tokenizerPath = tokenizerPath,
        _dimension = dimension;

  @override
  int get dimension => _dimension;

  @override
  Future<void> init() async {
    if (_initialized) return;

    // Load tokenizer
    _tokenizer = ClipTokenizer.fromFile(_tokenizerPath);

    // Load ONNX session
    final modelBytes = await File(_modelPath).readAsBytes();
    final sessionOptions = OrtSessionOptions();
    sessionOptions.appendDefaultProviders();
    _session = OrtSession.fromBuffer(modelBytes, sessionOptions);

    _initialized = true;
  }

  @override
  Future<Float32List> embed(String text) async {
    if (!_initialized || _session == null || _tokenizer == null) {
      throw StateError('ClipTextEmbedder not initialized. Call init() first.');
    }

    // Tokenize: text → [BOS] token_ids [EOS] [PAD]... (length 77)
    final tokenIds = _tokenizer!.encode(text);

    // Run inference
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      tokenIds,
      [1, ClipTokenizer.contextLength],
    );
    final runOptions = OrtRunOptions();
    final outputs = await _session!.runAsync(
      runOptions,
      {'input_ids': inputTensor},
    );
    inputTensor.release();
    runOptions.release();

    if (outputs == null || outputs.isEmpty || outputs.first == null) {
      throw StateError('CLIP text inference returned no output');
    }

    final outputValue = outputs.first!.value;
    outputs.first!.release();

    // Output is nested List<List<double>> with shape [1, 512]
    final flattened = _flattenToFloat32List(outputValue);
    l2Normalize(flattened);
    return flattened;
  }

  /// Release the ONNX session. Call when done (e.g. on idle timer).
  Future<void> dispose() async {
    await _session?.release();
    _session = null;
    _tokenizer = null;
    _initialized = false;
  }
}

// ─── CLIP Image Embedder ────────────────────────────────────────────────────

/// CLIP vision tower — embeds images into the same 512-dim space as text.
///
/// Used at ingest time to create image embeddings for cross-modal search.
class ClipImageEmbedder implements ImageEmbedder {
  final String _modelPath;
  final int _dimension;

  OrtSession? _session;
  bool _initialized = false;

  ClipImageEmbedder({
    required String modelPath,
    int dimension = 512,
  })  : _modelPath = modelPath,
        _dimension = dimension;

  @override
  int get dimension => _dimension;

  @override
  bool get isLoaded => _initialized;

  @override
  Future<void> init() async {
    if (_initialized) return;

    final modelBytes = await File(_modelPath).readAsBytes();
    final sessionOptions = OrtSessionOptions();
    sessionOptions.appendDefaultProviders();
    _session = OrtSession.fromBuffer(modelBytes, sessionOptions);

    _initialized = true;
  }

  @override
  Future<void> dispose() async {
    await _session?.release();
    _session = null;
    _initialized = false;
  }

  @override
  Future<Float32List> embedFile(String imagePath) async {
    if (!_initialized || _session == null) {
      throw StateError('ClipImageEmbedder not initialized. Call init() first.');
    }

    // Preprocess: read image, resize to 224x224, normalize, transpose to NCHW
    final pixelValues = await _preprocessImage(imagePath);

    // Run inference
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      pixelValues,
      [1, 3, 224, 224],
    );
    final runOptions = OrtRunOptions();
    final outputs = await _session!.runAsync(
      runOptions,
      {'pixel_values': inputTensor},
    );
    inputTensor.release();
    runOptions.release();

    if (outputs == null || outputs.isEmpty || outputs.first == null) {
      throw StateError('CLIP vision inference returned no output');
    }

    final outputValue = outputs.first!.value;
    outputs.first!.release();

    final flattened = _flattenToFloat32List(outputValue);
    l2Normalize(flattened);
    return flattened;
  }

  /// Preprocess an image file for CLIP vision tower.
  ///
  /// Steps:
  ///   1. Decode image (PNG, JPEG, WebP, GIF first frame) via package:image
  ///   2. Resize to 224x224 (bicubic) — CLIP's expected input size
  ///   3. Convert to RGB (drop alpha)
  ///   4. Normalize: (pixel / 255 - mean) / std
  ///      mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
  ///   5. Transpose HWC → CHW (channel-first for ONNX)
  ///   6. Flatten to Float32List of length 3*224*224 = 150528
  Future<Float32List> _preprocessImage(String imagePath) async {
    // Decode the image. package:image handles PNG/JPEG/WebP/GIF/etc.
    // For animated GIFs, decodeImage returns the first frame by default.
    final bytes = await File(imagePath).readAsBytes();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) {
      throw StateError('Could not decode image: $imagePath');
    }

    // Resize to 224x224 using bicubic interpolation (CLIP standard).
    // Maintains aspect ratio by squashing — CLIP doesn't do center crop
    // in the ONNX model, so we do it here.
    final resized = img.copyResize(
      decoded,
      width: 224,
      height: 224,
      interpolation: img.Interpolation.cubic,
    );

    // CLIP normalization constants (ImageNet mean/std)
    const meanR = 0.485, meanG = 0.456, meanB = 0.406;
    const stdR = 0.229, stdG = 0.224, stdB = 0.225;

    // Output: CHW float32, length 3*224*224 = 150528
    final pixelValues = Float32List(3 * 224 * 224);

    // Iterate pixels in HWC order, write to CHW order
    for (var y = 0; y < 224; y++) {
      for (var x = 0; x < 224; x++) {
        final pixel = resized.getPixel(x, y);
        // getPixel returns a Pixel with .r .g .b .a (normalized 0-1 in v4.x)
        // Convert to 0-1 range if needed
        final r = pixel.r.toDouble();
        final g = pixel.g.toDouble();
        final b = pixel.b.toDouble();

        // Normalize: (x - mean) / std
        final hwcIdx = y * 224 + x;
        pixelValues[0 * 224 * 224 + hwcIdx] = ((r - meanR) / stdR).toDouble();
        pixelValues[1 * 224 * 224 + hwcIdx] = ((g - meanG) / stdG).toDouble();
        pixelValues[2 * 224 * 224 + hwcIdx] = ((b - meanB) / stdB).toDouble();
      }
    }

    return pixelValues;
  }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Recursively flatten a nested List to a Float32List.
///
/// OrtValueTensor.value returns nested lists matching the output shape:
///   - shape [1, 512] → List<List<double>>
///   - shape [1, 1, 960, 960] → 4-level nested
Float32List _flattenToFloat32List(dynamic value) {
  final flat = <double>[];
  void recurse(dynamic v) {
    if (v is List) {
      for (final item in v) {
        recurse(item);
      }
    } else if (v is num) {
      flat.add(v.toDouble());
    }
  }

  recurse(value);
  return Float32List.fromList(flat);
}

// ─── Model2Vec Embedder ─────────────────────────────────────────────────────

/// model2vec text embedder — fast static embeddings for pure-text search.
///
/// Uses potion-base-32M (512-dim) from minishlab. Much faster than CLIP
/// text tower for query embedding (~1ms vs ~70ms). Used for the main
/// text search path. CLIP text tower is reserved for cross-modal queries
/// (text → image search).
///
/// model2vec uses Native Assets + Rust FFI. Requires Rust toolchain at
/// build time. The native library is bundled automatically.
class Model2VecEmbedder implements TextEmbedder {
  final String _modelPath;
  final int _dimension;

  bool _initialized = false;

  Model2VecEmbedder({
    required String modelPath,
    int dimension = 512,
  })  : _modelPath = modelPath,
        _dimension = dimension;

  @override
  int get dimension => _dimension;

  @override
  Future<void> init() async {
    if (_initialized) return;

    // Model2Vec.instance is a singleton. initEmbedder takes a HuggingFace
    // repo ID or a local directory path. We pass the local path.
    //
    // The model dir should contain:
    //   - model.safetensors
    //   - tokenizer.json
    //   - tokenizer_config.json
    //   - special_tokens_map.json
    //   - config.json
    Model2Vec.instance.initEmbedder(_modelPath);

    _initialized = true;
  }

  @override
  Future<Float32List> embed(String text) async {
    if (!_initialized) {
      throw StateError('Model2VecEmbedder not initialized. Call init() first.');
    }

    // generateEmbeddingAsync runs in a background isolate — doesn't block UI.
    // model2vec already L2-normalizes by default (isNormalized returns true),
    // so we don't need to call l2Normalize again.
    final embedding = await Model2Vec.instance.generateEmbeddingAsync(text);

    // Verify dimension matches what we expect
    if (embedding.length != _dimension) {
      throw StateError(
          'Model2Vec returned ${embedding.length}-dim vector, expected $_dimension');
    }

    return embedding;
  }

  /// Batch embed multiple texts at once. More efficient than calling embed()
  /// in a loop — model2vec uses Rust SIMD for batch processing.
  Future<List<Float32List>> embedBatch(List<String> texts) async {
    if (!_initialized) {
      throw StateError('Model2VecEmbedder not initialized. Call init() first.');
    }

    return await Model2Vec.instance.generateBatchEmbeddingsAsync(texts);
  }
}
