import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:forui/forui.dart';
import 'package:signals_flutter/signals_flutter.dart';
import 'package:path/path.dart' as p;

import 'package:meme_collector_core/meme_collector_core.dart';

import '../state/signals.dart';

class GifGrid extends StatelessWidget {
  const GifGrid({super.key});

  @override
  Widget build(BuildContext context) {
    return SignalBuilder(
      builder: (context) {
        // Empty query -> hotlist; otherwise search results.
        final query = searchQuery.value;
        final items = query.isEmpty ? recentUsage.value : searchResults.value;

        if (items.isEmpty && query.isNotEmpty) {
          return Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              spacing: 8,
              children: [
                const Icon(FLucideIcons.searchX, size: 32),
                Text(isSearching.value ? 'Searching…' : 'No matches'),
              ],
            ),
          );
        }

        if (items.isEmpty) {
          return Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              spacing: 12,
              children: [
                const Icon(FLucideIcons.imagePlus, size: 40),
                const Text('Your library is empty'),
                FButton(
                  mainAxisSize: MainAxisSize.min,
                  onPress: () {
                    // TODO: trigger import flow (paste URL dialog)
                  },
                  child: const Text('Add your first reaction'),
                ),
              ],
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
            // items is List<SearchResult> — look up the Reaction
            final result = items[i];
            final reaction = coordinator?.getReaction(result.reactionId);
            if (reaction == null) {
              return const SizedBox.shrink();
            }
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

    // Resolve thumbnail path (relative to storage root)
    String? thumbnailAbsPath;
    if (reaction.thumbnailStatic != null) {
      // storage root is parent of the metadata.json
      // We need to resolve relative paths against the storage root
      // For now, use a hack: coordinator stores reactions with relative paths,
      // we need the absolute path for Image.file
      // TODO: expose storage root path via coordinator for path resolution
      // For now, just show placeholder if we can't resolve
    }

    return SignalBuilder(
      builder: (context) {
        final progress = ingestProgress.value[reaction.id];

        return FCard.raw(
          clipBehavior: Clip.antiAlias,
          child: InkWell(
            onTap: () async {
              // Copy URL to clipboard (the Discord-like behavior)
              await Clipboard.setData(ClipboardData(text: reaction.url));
              await c.incrementUsage(reaction.id);
              if (context.mounted) {
                FToaster.show(context, message: 'Link copied');
              }
            },
            onLongPress: () {
              // TODO: right-click context menu (copy URL / file path / bytes / etc.)
            },
            child: Stack(
              fit: StackFit.expand,
              children: [
                // Thumbnail or placeholder
                thumbnailAbsPath != null
                    ? Image.file(File(thumbnailAbsPath), fit: BoxFit.cover)
                    : Container(
                        color: colors.muted,
                        child: reaction.status == ReactionStatus.queued ||
                                reaction.status == ReactionStatus.downloading
                            ? Icon(FLucideIcons.clock, size: 28, color: colors.mutedForeground)
                            : reaction.status == ReactionStatus.failed
                                ? Icon(FLucideIcons.alertCircle,
                                    size: 28, color: colors.destructive)
                                : Icon(FLucideIcons.image,
                                    size: 28, color: colors.mutedForeground),
                      ),
                // Usage count badge
                if (reaction.usageCount > 0)
                  Positioned(
                    right: 4,
                    bottom: 4,
                    child: FBadge(
                      variant: FBadgeVariant.secondary,
                      child: Text('${reaction.usageCount}'),
                    ),
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
