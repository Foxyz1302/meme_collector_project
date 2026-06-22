import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:forui/forui.dart';
import 'package:signals_flutter/signals_flutter.dart';
import 'package:waterfall_flow/waterfall_flow.dart';

import 'package:meme_collector_core/meme_collector_core.dart';

import '../state/signals.dart';

class GifGrid extends StatelessWidget {
  const GifGrid({super.key});

  @override
  Widget build(BuildContext context) {
    return SignalBuilder(
      builder: (context) {
        final query = searchQuery.value;
        // When searching: show search results (List<SearchResult>)
        // When not searching: show allReactions (List<Reaction>) so newly
        // added reactions appear immediately without waiting for hotlist refresh
        final searchItems = query.isEmpty ? <SearchResult>[] : searchResults.value;
        final allItems = allReactions.value;

        if (query.isEmpty) {
          // Show all reactions, sorted by addedAt desc (newest first)
          if (allItems.isEmpty) {
            return Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                spacing: 12,
                children: [
                  const Icon(FLucideIcons.imagePlus, size: 40),
                  const Text('Your library is empty'),
                  FButton(
                    mainAxisSize: MainAxisSize.min,
                    onPress: () async {
                      final clipData = await Clipboard.getData('text/plain');
                      final url = clipData?.text?.trim();
                      if (url != null && url.isNotEmpty) {
                        await addReaction(url);
                      }
                    },
                    child: const Text('Paste reaction URL'),
                  ),
                ],
              ),
            );
          }
          return CustomScrollView(
            slivers: [
              SliverWaterfallFlow(
                gridDelegate: SliverWaterfallFlowDelegateWithMaxCrossAxisExtent(
                  maxCrossAxisExtent: 300,
                  crossAxisSpacing: 4,
                  mainAxisSpacing: 4,
                ),
                delegate: SliverChildBuilderDelegate((context, index) {
                  return _GifTile(reaction: allItems[index]);
                }, childCount: allItems.length),
              ),
              // SliverWaterfallFlow(
              //   gridDelegate: const SliverWaterfallFlowDelegateWithMaxCrossAxisExtent(
              //     maxCrossAxisExtent: 200,
              //     crossAxisSpacing: 4,
              //     mainAxisSpacing: 4,
              //   ),
              //   delegate: SliverChildBuilderDelegate((context, i) {
              //     return Image(image: NetworkImage(allItems[i].url));
              //   }, childCount: allItems.length),
              // ),
            ],
          );
          // GridView.builder(
          //   padding: const EdgeInsets.all(8),
          //   gridDelegate: const SliverGridDelegateWithMaxCrossAxisExtent(
          //     maxCrossAxisExtent: 180,
          //     mainAxisSpacing: 8,
          //     crossAxisSpacing: 8,
          //     childAspectRatio: 1.0,
          //   ),
          //   itemCount: allItems.length,
          //   itemBuilder: (context, i) => _GifTile(reaction: allItems[i]),
          // );
        }

        // Search mode — show search results
        final items = searchItems;
        if (items.isEmpty) {
          return Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              spacing: 8,
              children: [const Icon(FLucideIcons.searchX, size: 32), Text(isSearching.value ? 'Searching…' : 'No matches')],
            ),
          );
        }

        return GridView.builder(
          padding: const EdgeInsets.all(8),
          gridDelegate: const SliverGridDelegateWithMaxCrossAxisExtent(
            maxCrossAxisExtent: 180,
            mainAxisSpacing: 8,
            crossAxisSpacing: 8,
            childAspectRatio: 1.0,
          ),
          itemCount: items.length,
          itemBuilder: (context, i) {
            final result = items[i];
            final reaction = coordinator?.getReaction(result.reactionId);
            if (reaction == null) return const SizedBox.shrink();
            return _GifTile(reaction: reaction);
          },
        );
      },
    );
  }
}

class _GifTile extends StatelessWidget {
  final Reaction reaction;
  const _GifTile({required this.reaction});

  @override
  Widget build(BuildContext context) {
    final colors = context.theme.colors;
    final c = coordinator!;

    // Resolve thumbnail path (relative to storage root) to absolute path
    final thumbnailAbsPath = reaction.thumbnailStatic != null ? c.resolvePath(reaction.thumbnailStatic!) : null;

    return SignalBuilder(
      builder: (context) {
        final progress = ingestProgress.value[reaction.id];

        return FTappable(
          onPress: () async {
            // Copy URL to clipboard (the Discord-like behavior)
            await Clipboard.setData(ClipboardData(text: reaction.url));
            await incrementUsage(reaction.id);
            if (context.mounted) {
              showFToast(context: context, title: const Text('Link copied'));
            }
          },
          onLongPress: () {
            // TODO: right-click context menu (copy URL / file path / bytes / etc.)
          },
          child: FCard.raw(
            clipBehavior: Clip.antiAlias,
            child: Stack(
              // fit: StackFit.expand,
              children: [
                // Thumbnail or placeholder
                thumbnailAbsPath != null
                    ? SizedBox(
                        width: .infinity,
                        child: Image.file(File(thumbnailAbsPath), fit: BoxFit.cover),
                      )
                    : Container(
                        height: 200,
                        width: .infinity,
                        color: colors.muted,
                        child: reaction.status == ReactionStatus.queued || reaction.status == ReactionStatus.downloading
                            ? Icon(FLucideIcons.clock, size: 28, color: colors.mutedForeground)
                            : reaction.status == ReactionStatus.failed
                            ? Icon(FLucideIcons.alertCircle, size: 28, color: colors.destructive)
                            : Icon(FLucideIcons.image, size: 28, color: colors.mutedForeground),
                      ),
                // Usage count badge
                if (reaction.usageCount > 0)
                  Positioned(
                    right: 4,
                    bottom: 4,
                    child: FBadge(variant: FBadgeVariant.secondary, child: Text('${reaction.usageCount}')),
                  ),
                // Ingest progress bar overlay
                if (progress != null && progress < 1.0)
                  Positioned(
                    left: 0,
                    right: 0,
                    bottom: 0,
                    child: LinearProgressIndicator(
                      value: progress,
                      minHeight: 3,
                      backgroundColor: colors.muted,
                      valueColor: AlwaysStoppedAnimation(colors.primary),
                    ),
                  ),
              ],
            ),
          ),
        );
      },
    );
  }
}
