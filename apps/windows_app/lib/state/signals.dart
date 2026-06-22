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

/// Live ingest progress: map of reactionId → progress (0.0–1.0).
/// Updated in real time as the ingest pipeline runs.
final ingestProgress = signal<Map<String, double>>({});

/// Initialize the coordinator and wire its streams to signals.
///
/// Call this once on startup, after the coordinator is created.
void wireCoordinatorToSignals() {
  final c = coordinator;
  if (c == null) return;

  // Live ingest progress → signal
  c.progressStream.listen((msg) {
    final progress = Map<String, double>.from(ingestProgress.value);
    progress[msg.reactionId] = msg.progress;
    ingestProgress.value = progress;
  });

  // When ingest completes, refresh allReactions + clear progress for that id
  c.completeStream.listen((event) {
    final progress = Map<String, double>.from(ingestProgress.value);
    progress.remove(event.reaction.id);
    ingestProgress.value = progress;
    allReactions.value = c.allReactions;
  });

  // When ingest fails, clear progress for that id
  c.failedStream.listen((msg) {
    final progress = Map<String, double>.from(ingestProgress.value);
    progress.remove(msg.reactionId);
    ingestProgress.value = progress;
    allReactions.value = c.allReactions;
  });
}

/// Refresh the allReactions signal from the coordinator.
void refreshReactions() {
  final c = coordinator;
  if (c == null) return;
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
