/// Unit tests for the search service.
///
/// Run from packages/meme_collector_core/:
///   dart test test/search_test.dart
///
/// Tests the in-memory vector index + keyword index + RRF fusion without
/// needing ONNX or real model embeddings. Uses synthetic vectors that are
/// constructed to have known similarity relationships.

library;

import 'dart:typed_data';

import 'package:meme_collector_core/meme_collector_core.dart';
import 'package:test/test.dart';

void main() {
  group('VectorIndex', () {
    test('add and search text vectors', () {
      final index = VectorIndex();

      // Create 3 synthetic vectors with known relationships:
      //   catVec and dogVec are similar (both animals)
      //   carVec is different
      final catVec = _makeVector([1.0, 0.1, 0.0, 0.0]);
      final dogVec = _makeVector([0.9, 0.2, 0.0, 0.0]); // similar to cat
      final carVec = _makeVector([0.0, 0.0, 0.0, 1.0]); // different

      index.addTextVector('cat-reaction', catVec);
      index.addTextVector('dog-reaction', dogVec);
      index.addTextVector('car-reaction', carVec);

      // Search with a vector similar to cat
      final queryVec = _makeVector([0.95, 0.15, 0.0, 0.0]);
      final results = index.searchText(queryVec, topK: 3);

      expect(results.length, 3);
      // cat-reaction should be #1 (most similar to query)
      expect(results[0].$1, 'cat-reaction');
      // dog-reaction should be #2 (also similar)
      expect(results[1].$1, 'dog-reaction');
      // car-reaction should be #3 (least similar)
      expect(results[2].$1, 'car-reaction');
    });

    test('remove reaction from index', () {
      final index = VectorIndex();

      index.addTextVector('a', _makeVector([1.0, 0.0]));
      index.addTextVector('b', _makeVector([0.0, 1.0]));
      index.addTextVector('c', _makeVector([1.0, 1.0]));

      expect(index.length, 3);

      index.remove('b');
      expect(index.length, 2);

      // Search should still work
      final results = index.searchText(_makeVector([1.0, 0.0]), topK: 5);
      expect(results.length, 2);
      expect(results.any((r) => r.$1 == 'b'), false);
    });

    test('image vectors are separate from text vectors', () {
      final index = VectorIndex();

      // Add a text vector but no image vector for 'a'
      index.addTextVector('a', _makeVector([1.0, 0.0]));
      // Add an image vector but no text vector for 'b'
      index.addImageVector('b', _makeVector([0.0, 1.0]));

      // Text search should only find 'a'
      final textResults = index.searchText(_makeVector([1.0, 0.0]), topK: 5);
      expect(textResults.length, 1);
      expect(textResults[0].$1, 'a');

      // Image search should only find 'b'
      final imageResults = index.searchImage(_makeVector([0.0, 1.0]), topK: 5);
      expect(imageResults.length, 1);
      expect(imageResults[0].$1, 'b');
    });
  });

  group('KeywordIndex', () {
    test('basic keyword matching', () {
      final index = KeywordIndex();

      index.add(_makeReaction(
        id: 'r1',
        title: 'drake hotline bling',
        tags: ['sarcastic', 'drake'],
      ));
      index.add(_makeReaction(
        id: 'r2',
        title: 'cat sitting on table',
        tags: ['cat', 'cute'],
      ));
      index.add(_makeReaction(
        id: 'r3',
        title: 'drake typing on keyboard',
        tags: ['drake', 'typing'],
      ));

      // Search for 'drake' — should match r1 and r3
      final results = index.search('drake', topK: 10);
      expect(results.length, 2);
      expect(results.any((r) => r.$1 == 'r1'), true);
      expect(results.any((r) => r.$1 == 'r3'), true);
      expect(results.any((r) => r.$1 == 'r2'), false);
    });

    test('multiple word matches rank higher', () {
      final index = KeywordIndex();

      index.add(_makeReaction(
        id: 'r1',
        title: 'drake hotline',
        tags: ['drake'],
      ));
      index.add(_makeReaction(
        id: 'r2',
        title: 'drake only',
        tags: ['other'],
      ));

      // Search for 'drake hotline' — r1 matches 2 words, r2 matches 1
      final results = index.search('drake hotline', topK: 10);
      expect(results.length, 2);
      expect(results[0].$1, 'r1'); // 2 matches → higher score
      expect(results[0].$2, 2.0);
      expect(results[1].$1, 'r2'); // 1 match
      expect(results[1].$2, 1.0);
    });

    test('ocr text is indexed', () {
      final index = KeywordIndex();

      index.add(_makeReaction(
        id: 'r1',
        title: 'meme',
        tags: [],
        ocrText: 'THIS IS FINE',
      ));

      // Search for 'fine' — should match via OCR text
      final results = index.search('fine', topK: 10);
      expect(results.length, 1);
      expect(results[0].$1, 'r1');
    });

    test('case insensitive', () {
      final index = KeywordIndex();

      index.add(_makeReaction(
        id: 'r1',
        title: 'Drake Hotline',
        tags: ['Sarcastic'],
      ));

      // Search with lowercase
      expect(index.search('drake', topK: 10).length, 1);
      expect(index.search('DRake', topK: 10).length, 1);
      expect(index.search('SARCASTIC', topK: 10).length, 1);
    });
  });

  group('SearchService', () {
    test('RRF fusion combines text + keyword results', () {
      final vectorIndex = VectorIndex();
      final keywordIndex = KeywordIndex();

      // Set up: 3 reactions with text vectors
      // r1: "drake hotline" — vector close to query
      // r2: "drake typing" — vector also close to query
      // r3: "cat sitting" — vector far from query
      vectorIndex.addTextVector('r1', _makeVector([0.9, 0.1]));
      vectorIndex.addTextVector('r2', _makeVector([0.85, 0.15]));
      vectorIndex.addTextVector('r3', _makeVector([0.0, 1.0]));

      keywordIndex.add(_makeReaction(id: 'r1', title: 'drake hotline', tags: []));
      keywordIndex.add(_makeReaction(id: 'r2', title: 'drake typing', tags: []));
      keywordIndex.add(_makeReaction(id: 'r3', title: 'cat sitting', tags: []));

      final reactionsById = {
        'r1': _makeReaction(id: 'r1', title: 'drake hotline', usageCount: 0),
        'r2': _makeReaction(id: 'r2', title: 'drake typing', usageCount: 5),
        'r3': _makeReaction(id: 'r3', title: 'cat sitting', usageCount: 0),
      };

      final service = SearchService(
        vectorIndex: vectorIndex,
        keywordIndex: keywordIndex,
      );

      final results = service.search(
        query: 'drake',
        textQueryVec: _makeVector([0.9, 0.1]),
        clipQueryVec: null,
        reactionsById: reactionsById,
        topK: 10,
      );

      // r1 and r2 should be top 2 (both match text vector + keyword 'drake')
      // r3 should be last (low vector similarity, no keyword match)
      // All 3 are returned because they're all in the vector index —
      // the search returns ranked results, the caller decides how many to show.
      expect(results.length, 3);
      expect(results[0].reactionId, anyOf('r1', 'r2'));
      expect(results[1].reactionId, anyOf('r1', 'r2'));
      expect(results[2].reactionId, 'r3'); // last = lowest score
    });

    test('usage boost affects ranking', () {
      final vectorIndex = VectorIndex();
      final keywordIndex = KeywordIndex();

      // Two reactions with identical vectors — only difference is usage count
      vectorIndex.addTextVector('r1', _makeVector([1.0, 0.0]));
      vectorIndex.addTextVector('r2', _makeVector([1.0, 0.0]));

      final reactionsById = {
        'r1': _makeReaction(id: 'r1', title: 'a', usageCount: 0),
        'r2': _makeReaction(id: 'r2', title: 'b', usageCount: 100),
      };

      final service = SearchService(
        vectorIndex: vectorIndex,
        keywordIndex: keywordIndex,
      );

      final results = service.search(
        query: 'test',
        textQueryVec: _makeVector([1.0, 0.0]),
        clipQueryVec: null,
        reactionsById: reactionsById,
        topK: 10,
      );

      // Both have same RRF score, but r2 has higher usage boost
      expect(results.length, 2);
      expect(results[0].reactionId, 'r2'); // higher usage → higher final score
      expect(results[0].usageBoost, greaterThan(results[1].usageBoost));
    });

    test('hotlist returns most-used reactions', () {
      final vectorIndex = VectorIndex();
      final keywordIndex = KeywordIndex();

      final reactionsById = {
        'r1': _makeReaction(id: 'r1', title: 'a', usageCount: 10),
        'r2': _makeReaction(id: 'r2', title: 'b', usageCount: 50),
        'r3': _makeReaction(id: 'r3', title: 'c', usageCount: 5),
      };

      final service = SearchService(
        vectorIndex: vectorIndex,
        keywordIndex: keywordIndex,
      );

      final results = service.hotlist(
        reactionsById: reactionsById,
        limit: 10,
      );

      expect(results.length, 3);
      expect(results[0].reactionId, 'r2'); // 50 uses
      expect(results[1].reactionId, 'r1'); // 10 uses
      expect(results[2].reactionId, 'r3'); // 5 uses
    });
  });
}

// ─── Test helpers ───────────────────────────────────────────────────────────

/// Create a normalized Float32List from a list of doubles.
Float32List _makeVector(List<double> values) {
  final vec = Float32List.fromList(values);
  l2Normalize(vec);
  return vec;
}

/// Create a Reaction for testing.
Reaction _makeReaction({
  required String id,
  String? title,
  List<String> tags = const [],
  String? ocrText,
  int usageCount = 0,
}) {
  return Reaction(
    id: id,
    url: 'https://example.com/$id',
    urlNormalized: 'test:$id',
    sourcePlatform: SourcePlatform.direct,
    addedAt: DateTime.now(),
    status: ReactionStatus.ready,
    title: title,
    tags: tags,
    ocrText: ocrText,
    usageCount: usageCount,
  );
}
