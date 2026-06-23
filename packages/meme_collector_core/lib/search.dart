/// In-memory vector index + keyword index + RRF fusion search.
///
/// Pure Dart — no Flutter deps, no ONNX. This runs in the search isolate
/// and owns the in-memory vector index.
///
/// Three retrievers fused via Reciprocal Rank Fusion (RRF):
///   1. Text vector ANN — semantic text search (model2vec embeddings)
///   2. Image vector ANN — cross-modal search (CLIP text → CLIP image)
///   3. Keyword search — exact word matching across title/tags/ocrText
///
/// RRF formula:
///   score(doc) = Σ 1 / (k + rank_i(doc))  for each retriever i
///   k = 60 (standard)
///
/// Plus a usage boost:
///   finalScore = rrf + 0.05 * log(1 + usageCount)
library;

import 'dart:collection';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'models.dart';
import 'storage.dart';

// ─── Vector Index ───────────────────────────────────────────────────────────

/// In-memory vector index for fast SIMD cosine similarity search.
///
/// Two parallel arrays — text vectors and image vectors — keyed by reaction ID.
/// Loads all vectors into memory at startup (~20 MB for 10k × 512-dim × 4 bytes).
///
/// Uses Float32x4 SIMD for 4× speedup on cosine similarity. Vectors are
/// pre-normalized at ingest time, so cosine sim = dot product (no sqrt needed).
class VectorIndex {
  /// Reaction ID → index in the parallel arrays.
  final Map<String, int> _idToIndex = {};

  /// Parallel arrays. Index i in all three refers to the same reaction.
  final List<String> _ids = [];
  final List<Float32List?> _textVectors = [];
  final List<Float32List?> _imageVectors = [];

  bool _loaded = false;

  /// Load all vectors from disk into memory.
  ///
  /// Scans the embeddings_text/ and embeddings_image/ directories for
  /// .f32.bin files. The directory name encodes the model version:
  ///   embeddings_text/model2vec__potion-base-32M/{id}.f32.bin
  ///   embeddings_image/clip-vit-b32-int8__v1/{id}.f32.bin
  ///
  /// Only loads vectors matching the [activeTextVersion] and [activeImageVersion].
  Future<void> load(
    Storage storage,
    String activeTextVersion,
    String activeImageVersion,
  ) async {
    _idToIndex.clear();
    _ids.clear();
    _textVectors.clear();
    _imageVectors.clear();

    // Load text vectors — producer must match what the ingest pipeline writes.
    // Currently using CLIP text tower (model2vec was removed).
    // Try multiple producer names for backward compat.
    for (final textProducer in ['clip-vit-b32-fp16', 'model2vec']) {
      final textDirName = '${textProducer}__$activeTextVersion';
      final textDir = Directory('${storage.embeddingsTextDir}/$textDirName');
      if (await textDir.exists()) {
        await for (final entry in textDir.list()) {
          if (entry is! File) continue;
          if (!entry.path.endsWith('.f32.bin')) continue;

          final id = entry.uri.pathSegments.last.replaceAll('.f32.bin', '');
          final vector = readVectorFile(entry.path);
          final index = _ids.length;
          _ids.add(id);
          _textVectors.add(vector);
          _imageVectors.add(null);
          _idToIndex[id] = index;
        }
        break; // found text vectors, stop searching
      }
    }

    // Load image vectors — try multiple producer names for backward compat.
    for (final imageProducer in ['clip-vit-b32-fp16', 'clip-vit-b32-int8']) {
      final imageDirName = '${imageProducer}__$activeImageVersion';
      final imageDir = Directory('${storage.embeddingsImageDir}/$imageDirName');
      if (await imageDir.exists()) {
        await for (final entry in imageDir.list()) {
          if (entry is! File) continue;
          if (!entry.path.endsWith('.f32.bin')) continue;

          final id = entry.uri.pathSegments.last.replaceAll('.f32.bin', '');
          final vector = readVectorFile(entry.path);

          final index = _idToIndex[id];
          if (index != null) {
            _imageVectors[index] = vector;
          } else {
            final newIndex = _ids.length;
            _ids.add(id);
            _textVectors.add(null);
            _imageVectors.add(vector);
            _idToIndex[id] = newIndex;
          }
        }
        break; // found image vectors, stop searching
      }
    }

    _loaded = true;
  }

  /// Add a text vector for a reaction. Called when ingest completes.
  void addTextVector(String reactionId, Float32List vector) {
    var index = _idToIndex[reactionId];
    if (index == null) {
      index = _ids.length;
      _ids.add(reactionId);
      _textVectors.add(vector);
      _imageVectors.add(null);
      _idToIndex[reactionId] = index;
    } else {
      _textVectors[index] = vector;
    }
  }

  /// Add an image vector for a reaction. Called when ingest completes.
  void addImageVector(String reactionId, Float32List vector) {
    var index = _idToIndex[reactionId];
    if (index == null) {
      index = _ids.length;
      _ids.add(reactionId);
      _textVectors.add(null);
      _imageVectors.add(vector);
      _idToIndex[reactionId] = index;
    } else {
      _imageVectors[index] = vector;
    }
  }

  /// Remove a reaction from the index (on delete).
  void remove(String reactionId) {
    final index = _idToIndex[reactionId];
    if (index == null) return;

    // Swap-remove: move last element to the removed slot, then shrink.
    final lastIndex = _ids.length - 1;
    if (index != lastIndex) {
      final lastId = _ids[lastIndex];
      _ids[index] = lastId;
      _textVectors[index] = _textVectors[lastIndex];
      _imageVectors[index] = _imageVectors[lastIndex];
      _idToIndex[lastId] = index;
    }
    _ids.removeLast();
    _textVectors.removeLast();
    _imageVectors.removeLast();
    _idToIndex.remove(reactionId);
  }

  /// Search text vectors for the nearest neighbors to [query].
  ///
  /// Returns (reactionId, score) pairs sorted by score descending.
  /// Score is cosine similarity (since vectors are pre-normalized, = dot product).
  /// Results below [minScore] are filtered out (default 0.15 — below that is
  /// essentially random for CLIP 512-dim embeddings).
  List<(String, double)> searchText(Float32List query, {int topK = 50, double minScore = 0.15}) {
    final results = <(String, double)>[];
    for (var i = 0; i < _ids.length; i++) {
      final vec = _textVectors[i];
      if (vec == null) continue;
      final score = _dotProductSimd(query, vec);
      if (score >= minScore) {
        results.add((_ids[i], score));
      }
    }
    results.sort((a, b) => b.$2.compareTo(a.$2));
    return results.take(topK).toList();
  }

  /// Search image vectors for the nearest neighbors to [query].
  ///
  /// [query] is typically a CLIP text embedding (cross-modal search).
  /// Results below [minScore] are filtered out.
  List<(String, double)> searchImage(Float32List query, {int topK = 50, double minScore = 0.15}) {
    final results = <(String, double)>[];
    for (var i = 0; i < _ids.length; i++) {
      final vec = _imageVectors[i];
      if (vec == null) continue;
      final score = _dotProductSimd(query, vec);
      if (score >= minScore) {
        results.add((_ids[i], score));
      }
    }
    results.sort((a, b) => b.$2.compareTo(a.$2));
    return results.take(topK).toList();
  }

  /// Number of reactions in the index.
  int get length => _ids.length;

  /// True if load() has been called.
  bool get isLoaded => _loaded;

  /// SIMD-optimized dot product using Float32x4.
  ///
  /// Vectors are pre-normalized so dot product = cosine similarity.
  /// Falls back to scalar if vector length isn't divisible by 4.
  static double _dotProductSimd(Float32List a, Float32List b) {
    if (a.length != b.length) {
      throw ArgumentError('Vector length mismatch: ${a.length} vs ${b.length}');
    }

    final len = a.length;
    final simdLen = len - (len % 4); // largest multiple of 4 ≤ len

    // SIMD path — 4 floats at a time
    if (simdLen > 0) {
      final a4 = Float32x4List.sublistView(a, 0, simdLen);
      final b4 = Float32x4List.sublistView(b, 0, simdLen);
      var sum4 = Float32x4.zero();
      for (var i = 0; i < a4.length; i++) {
        sum4 += a4[i] * b4[i];
      }
      // Sum the 4 lanes
      var dot = sum4.x + sum4.y + sum4.z + sum4.w;

      // Tail (remaining 0-3 elements)
      for (var i = simdLen; i < len; i++) {
        dot += a[i] * b[i];
      }
      return dot.toDouble();
    }

    // Scalar fallback for very short vectors
    var dot = 0.0;
    for (var i = 0; i < len; i++) {
      dot += a[i] * b[i];
    }
    return dot;
  }
}

// ─── Keyword Index ──────────────────────────────────────────────────────────

/// Simple inverted index for exact word matching.
///
/// Builds a `Map<word, Set<reactionId>>` from title + tags + ocrText.
/// Search returns reactions with at least one matching word, ranked by
/// match count (more matches = higher rank).
///
/// No stemming, no fuzzy matching. For 10k items this is fine — if you
/// need better later, swap in a proper FTS, the interface stays the same.
class KeywordIndex {
  final Map<String, Set<String>> _index = {};

  /// Rebuild the entire index from a list of reactions.
  void rebuild(List<Reaction> reactions) {
    _index.clear();
    for (final r in reactions) {
      _addReaction(r);
    }
  }

  /// Add a single reaction to the index.
  void add(Reaction r) {
    _addReaction(r);
  }

  /// Remove a reaction from the index.
  void remove(String reactionId) {
    // O(n) scan — fine for our scale
    for (final entry in _index.entries) {
      entry.value.remove(reactionId);
      if (entry.value.isEmpty) {
        // Will be cleaned up on next rebuild, or we could remove now
      }
    }
  }

  void _addReaction(Reaction r) {
    final words = _extractWords(r);
    for (final word in words) {
      _index.putIfAbsent(word, () => <String>{}).add(r.id);
    }
  }

  /// Extract lowercase words from title + tags + ocrText.
  Set<String> _extractWords(Reaction r) {
    final words = <String>{};
    final text = [
      r.title ?? '',
      ...r.tags,
      r.ocrText ?? '',
    ].join(' ');
    for (final match in RegExp(r'[a-z0-9]+').allMatches(text.toLowerCase())) {
      final word = match.group(0)!;
      if (word.length >= 2) {
        // Skip 1-char "words" (noise)
        words.add(word);
      }
    }
    return words;
  }

  /// Search for reactions matching any word in [query].
  ///
  /// Returns (reactionId, score) pairs where score = number of matching words.
  /// Sorted by score descending.
  List<(String, double)> search(String query, {int topK = 50}) {
    final queryWords = RegExp(r'[a-z0-9]+')
        .allMatches(query.toLowerCase())
        .map((m) => m.group(0)!)
        .where((w) => w.length >= 2)
        .toSet();

    if (queryWords.isEmpty) return [];

    // Count matches per reaction
    final matchCounts = HashMap<String, int>();
    for (final word in queryWords) {
      final matches = _index[word];
      if (matches == null) continue;
      for (final reactionId in matches) {
        matchCounts[reactionId] = (matchCounts[reactionId] ?? 0) + 1;
      }
    }

    final results = matchCounts.entries
        .map((e) => (e.key, e.value.toDouble()))
        .toList();
    results.sort((a, b) => b.$2.compareTo(a.$2));
    return results.take(topK).toList();
  }

  /// Number of unique words in the index.
  int get wordCount => _index.length;
}

// ─── Search Service ─────────────────────────────────────────────────────────

/// Orchestrates search across multiple retrievers using RRF fusion.
///
/// Pipeline:
///   1. Embed query with model2vec → textVec
///   2. (Optional) Embed query with CLIP text → clipVec (for image search)
///   3. Parallel retrieval:
///      - textHits = vectorIndex.searchText(textVec, topK=50)
///      - keywordHits = keywordIndex.search(query, topK=50)
///      - imageHits = vectorIndex.searchImage(clipVec, topK=50)  [if clipVec provided]
///   4. RRF fusion: score(d) = Σ 1/(60 + rank_i(d))
///   5. Usage boost: finalScore = rrf + 0.05 * log(1 + usageCount)
///   6. Return top K reaction IDs
class SearchService {
  final VectorIndex vectorIndex;
  final KeywordIndex keywordIndex;

  /// RRF k parameter. Standard value is 60.
  static const int _rrfK = 60;

  /// Usage boost factor. log(1 + usageCount) * this.
  static const double _usageBoost = 0.05;

  SearchService({
    required this.vectorIndex,
    required this.keywordIndex,
  });

  /// Search for reactions matching [query].
  ///
  /// [textQueryVec] — model2vec embedding of the query (required)
  /// [clipQueryVec] — CLIP text embedding of the query (optional, enables
  ///                  cross-modal image search)
  /// [reactionsById] — map of reactionId → Reaction, for usage count lookup
  /// [topK] — number of results to return
  List<SearchResult> search({
    required String query,
    required Float32List textQueryVec,
    Float32List? clipQueryVec,
    required Map<String, Reaction> reactionsById,
    int topK = 50,
  }) {
    if (query.isEmpty) return [];

    // Parallel retrieval
    final textHits = vectorIndex.searchText(textQueryVec, topK: 50);
    final keywordHits = keywordIndex.search(query, topK: 50);
    final List<(String, double)> imageHits;
    if (clipQueryVec != null) {
      imageHits = vectorIndex.searchImage(clipQueryVec, topK: 50);
    } else {
      imageHits = const [];
    }

    // RRF fusion
    final rrfScores = HashMap<String, double>();
    void addRrf(List<(String, double)> hits) {
      for (var rank = 0; rank < hits.length; rank++) {
        final id = hits[rank].$1;
        rrfScores[id] = (rrfScores[id] ?? 0) + 1.0 / (_rrfK + rank + 1);
      }
    }

    addRrf(textHits);
    addRrf(keywordHits);
    addRrf(imageHits);

    // Apply usage boost + collect results
    // Skip reactions not in reactionsById (stale vectors on disk for deleted reactions)
    final results = <SearchResult>[];
    for (final entry in rrfScores.entries) {
      final id = entry.key;
      final rrf = entry.value;
      final reaction = reactionsById[id];
      if (reaction == null) continue; // stale — vector exists but metadata doesn't
      final usageCount = reaction.usageCount;
      final usageBoost = _usageBoost * math.log(1 + usageCount);
      final score = rrf + usageBoost;
      results.add(SearchResult(
        reactionId: id,
        score: score,
        rrfScore: rrf,
        usageBoost: usageBoost,
      ));
    }

    // Sort by RRF score only (no usage boost — search should be predictable)
    results.sort((a, b) => b.rrfScore.compareTo(a.rrfScore));

    return results.take(topK).map((r) => SearchResult(
      reactionId: r.reactionId,
      score: r.rrfScore,
      rrfScore: r.rrfScore,
      usageBoost: 0,
    )).toList();
  }

  /// Get the hotlist — most recently used reactions.
  ///
  /// Used when search query is empty (the "default" view).
  List<SearchResult> hotlist({
    required Map<String, Reaction> reactionsById,
    int limit = 60,
  }) {
    final reactions = reactionsById.values.where((r) => r.status == ReactionStatus.ready).toList();
    reactions.sort((a, b) {
      // Sort by usage count desc, then by last used desc
      final usageCmp = b.usageCount.compareTo(a.usageCount);
      if (usageCmp != 0) return usageCmp;
      final aTime = a.lastUsedAt?.millisecondsSinceEpoch ?? 0;
      final bTime = b.lastUsedAt?.millisecondsSinceEpoch ?? 0;
      return bTime.compareTo(aTime);
    });

    return reactions.take(limit).map((r) => SearchResult(
      reactionId: r.id,
      score: r.usageCount.toDouble(),
      rrfScore: 0,
      usageBoost: 0,
    )).toList();
  }
}

// ─── Search Result ──────────────────────────────────────────────────────────

/// A single search result with scoring breakdown for debugging.
class SearchResult {
  final String reactionId;
  final double score; // final score (RRF + usage boost)
  final double rrfScore; // RRF component
  final double usageBoost; // usage boost component

  const SearchResult({
    required this.reactionId,
    required this.score,
    required this.rrfScore,
    required this.usageBoost,
  });

  @override
  String toString() =>
      'SearchResult($reactionId, score=$score, rrf=$rrfScore, boost=$usageBoost)';
}
