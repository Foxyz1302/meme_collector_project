/// Reactive state for the app.
///
/// These signals are the bridge between the Coordinator (core backend) and
/// the Forui widgets. Widgets use SignalBuilder to rebuild when signals change.
///
/// The Coordinator is set once on startup via `initCoordinator()`. After that,
/// widgets can access it via the global `coordinator` variable.
library;

import 'package:signals_flutter/signals_flutter.dart';
import 'package:meme_collector_core/meme_collector_core.dart';

/// The global Coordinator instance. Set once on startup.
/// Null before initCoordinator() is called.
Coordinator? coordinator;

/// Current search query (debounced by the search bar).
final searchQuery = signal('');

/// Whether a search is currently in progress.
final isSearching = signal(false);

/// Current search results (List of SearchResult, empty when no query).
final searchResults = signal<List<SearchResult>>([]);

/// Most-recently-used reactions (hotlist, shown when search is empty).
final recentUsage = signal<List<SearchResult>>([]);

/// All reactions in the library (for the grid when not searching).
final allReactions = signal<List<Reaction>>([]);

/// Currently selected nav item: 'all' | 'recent' | 'tags' | 'sources'
final selectedNav = signal('all');

/// Live ingest status text per reaction: map of reactionId → status text.
/// Shows what the pipeline is currently doing (e.g. "Downloading…",
/// "Generating thumbnail…", "Embedding…") instead of a progress bar.
final ingestStatus = signal<Map<String, String>>({});

/// Per-reaction dimensions captured during image load (before ingest runs).
/// Map of reactionId → (width, height). Used for masonry layout.
final reactionDimensions = signal<Map<String, (int, int)>>({});

/// Count of reactions with missing embeddings. Updated after init and
/// after each ingest completes. When > 0, sidebar shows a notice.
final incompleteCount = signal<int>(0);

/// Human-readable status text for a reaction's ingest pipeline state.
String _statusText(ReactionStatus status) {
  switch (status) {
    case ReactionStatus.queued:
      return 'Queued…';
    case ReactionStatus.downloading:
      return 'Downloading…';
    case ReactionStatus.thumbnailing:
      return 'Generating thumbnail…';
    case ReactionStatus.embedding:
      return 'Embedding…';
    case ReactionStatus.ocr:
      return 'Reading text…';
    case ReactionStatus.ready:
      return '';
    case ReactionStatus.failed:
      return 'Failed';
  }
}

/// Initialize the coordinator and wire its streams to signals.
///
/// Call this once on startup, after the coordinator is created.
void wireCoordinatorToSignals() {
  final c = coordinator;
  if (c == null) return;

  // Live ingest status → signal
  c.progressStream.listen((msg) {
    print('[Signals] progressStream: ${msg.reactionUrl} → ${_statusText(msg.status)}');
    final status = Map<String, String>.from(ingestStatus.value);
    status[msg.reactionUrl] = _statusText(msg.status);
    ingestStatus.value = status;
  });

  // When ingest completes, refresh allReactions + clear status + rescan incomplete
  c.completeStream.listen((event) {
    final status = Map<String, String>.from(ingestStatus.value);
    status.remove(event.reaction.id);
    ingestStatus.value = status;
    allReactions.value = c.allReactions;
    incompleteCount.value = c.findIncompleteReactions().length;
  });

  // When ingest fails, show error status + rescan incomplete
  c.failedStream.listen((msg) {
    final status = Map<String, String>.from(ingestStatus.value);
    status[msg.reactionId] = 'Failed: ${msg.error}';
    ingestStatus.value = status;
    allReactions.value = c.allReactions;
    incompleteCount.value = c.findIncompleteReactions().length;
  });
}

/// Refresh the allReactions signal from the coordinator.
void refreshReactions() {
  final c = coordinator;
  if (c == null) return;
  print('[Signals] refreshReactions: ${c.allReactions.length} reactions');
  allReactions.value = c.allReactions;
}

/// Refresh the hotlist (most-used reactions).
Future<void> refreshHotlist() async {
  final c = coordinator;
  if (c == null) return;
  recentUsage.value = await c.hotlist();
}

/// Run a search and update the searchResults signal.
Future<void> performSearch(String query) async {
  final c = coordinator;
  if (c == null) return;

  if (query.isEmpty) {
    searchResults.value = [];
    await refreshHotlist();
    return;
  }

  isSearching.value = true;
  try {
    searchResults.value = await c.search(query);
  } catch (e) {
    searchResults.value = [];
  } finally {
    isSearching.value = false;
  }
}

/// Add a reaction from a URL. Refreshes signals so the grid updates immediately.
/// The reaction appears in the grid instantly (with "Queued…" status), then
/// the ingest pipeline runs in the background and updates the tile progressively.
Future<void> addReaction(String url, {String? title, List<String>? tags}) async {
  final c = coordinator;
  if (c == null) return;

  // Add to coordinator (creates entry in metadata + starts ingest pipeline)
  // This returns quickly — the reaction is in metadata immediately with
  // status=queued, and the pipeline runs async in the background.
  await c.addReaction(url: url, title: title, tags: tags);

  // Refresh the grid signal so the new reaction appears immediately
  refreshReactions();
}

/// Increment usage for a reaction. Refreshes signals so the grid updates.
Future<void> incrementUsage(String id) async {
  final c = coordinator;
  if (c == null) return;
  await c.incrementUsage(id);
  refreshReactions();
}

/// Scan for reactions with missing embeddings. Returns count and updates signal.
int scanIncomplete() {
  final c = coordinator;
  if (c == null) return 0;
  final incomplete = c.findIncompleteReactions();
  incompleteCount.value = incomplete.length;
  return incomplete.length;
}

/// Process all incomplete reactions (missing embeddings).
/// Returns count queued.
int processIncomplete() {
  final c = coordinator;
  if (c == null) return 0;
  final count = c.processIncomplete();
  refreshReactions();
  return count;
}

/// Force re-embed ALL reactions. Returns count queued.
int resetAll() {
  final c = coordinator;
  if (c == null) return 0;
  final count = c.resetAll();
  refreshReactions();
  return count;
}
