import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:forui/forui.dart';
import 'package:signals_flutter/signals_flutter.dart';
import 'package:waterfall_flow/waterfall_flow.dart';

import 'package:meme_collector_core/meme_collector_core.dart';

import '../reaction_image_provider.dart';
import '../state/signals.dart';

class GifGrid extends StatelessWidget {
  const GifGrid({super.key});

  @override
  Widget build(BuildContext context) {
    return SignalBuilder(
      builder: (context) {
        final query = searchQuery.value;
        final searchItems =
            query.isEmpty ? <SearchResult>[] : searchResults.value;
        final allItems = allReactions.value;

        if (query.isEmpty) {
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
          final sortedItems = List<Reaction>.from(allItems)
            ..sort((a, b) => b.id.compareTo(a.id));
          return CustomScrollView(
            slivers: [
              SliverWaterfallFlow(
                gridDelegate:
                    SliverWaterfallFlowDelegateWithMaxCrossAxisExtent(
                  maxCrossAxisExtent: 300,
                  crossAxisSpacing: 4,
                  mainAxisSpacing: 4,
                ),
                delegate: SliverChildBuilderDelegate(
                  (context, index) => _GifTile(reaction: sortedItems[index]),
                  childCount: sortedItems.length,
                ),
              ),
            ],
          );
        }

        // Search mode
        final items = searchItems;
        if (items.isEmpty) {
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

        return CustomScrollView(
          slivers: [
            SliverWaterfallFlow(
              gridDelegate:
                  SliverWaterfallFlowDelegateWithMaxCrossAxisExtent(
                maxCrossAxisExtent: 300,
                crossAxisSpacing: 4,
                mainAxisSpacing: 4,
              ),
              delegate: SliverChildBuilderDelegate(
                (context, i) {
                  final result = items[i];
                  final reaction = coordinator?.getReaction(result.reactionId);
                  if (reaction == null) return const SizedBox.shrink();
                  return _GifTile(reaction: reaction);
                },
                childCount: items.length,
              ),
            ),
          ],
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

    final thumbnailStaticAbs = reaction.thumbnailStatic != null
        ? c.resolvePath(reaction.thumbnailStatic!)
        : null;
    final thumbnailAnimatedAbs = reaction.thumbnailAnimated != null
        ? c.resolvePath(reaction.thumbnailAnimated!)
        : null;
    final localFileAbs =
        reaction.localFile != null ? c.resolvePath(reaction.localFile!) : null;

    return FContextMenu(
      menu: [
        FContextMenuEntry.group(
          children: [
            FContextMenuEntry.item(
              prefix: const Icon(FLucideIcons.link, size: 16),
              title: const Text('Copy URL'),
              onPress: () async {
                await Clipboard.setData(ClipboardData(text: reaction.url));
                if (context.mounted) {
                  showFToast(context: context, title: const Text('Link copied'));
                }
              },
            ),
            if (localFileAbs != null)
              FContextMenuEntry.item(
                prefix: const Icon(FLucideIcons.fileCopy, size: 16),
                title: const Text('Copy file path'),
                onPress: () async {
                  await Clipboard.setData(ClipboardData(text: localFileAbs));
                  if (context.mounted) {
                    showFToast(context: context, title: const Text('Path copied'));
                  }
                },
              ),
            FContextMenuEntry.item(
              prefix: const Icon(FLucideIcons.externalLink, size: 16),
              title: const Text('Open in browser'),
              onPress: () {
                // TODO: url_launcher
              },
            ),
          ],
        ),
        FContextMenuEntry.group(
          children: [
            FContextMenuEntry.item(
              prefix: const Icon(FLucideIcons.refreshCw, size: 16),
              title: const Text('Re-embed'),
              onPress: () async {
                await c.reEmbedImage(reaction.id);
                if (context.mounted) {
                  showFToast(context: context, title: const Text('Re-embedded'));
                }
              },
            ),
            FContextMenuEntry.item(
              prefix: const Icon(FLucideIcons.imageRefresh, size: 16),
              title: const Text('Re-generate thumbnail'),
              onPress: () async {
                await c.regenerateThumbnails(reaction.id);
                refreshReactions();
                if (context.mounted) {
                  showFToast(context: context, title: const Text('Thumbnail regenerated'));
                }
              },
            ),
            if (!reaction.pinned)
              FContextMenuEntry.item(
                prefix: const Icon(FLucideIcons.pin, size: 16),
                title: const Text('Pin locally'),
                onPress: () async {
                  await c.pinReaction(reaction.id);
                  refreshReactions();
                },
              ),
          ],
        ),
        FContextMenuEntry.group(
          children: [
            FContextMenuEntry.item(
              prefix: const Icon(FLucideIcons.trash2, size: 16),
              title: const Text('Delete'),
              onPress: () async {
                await c.deleteReaction(reaction.id);
                refreshReactions();
                if (context.mounted) {
                  showFToast(context: context, title: const Text('Deleted'));
                }
              },
            ),
          ],
        ),
      ],
      child: FTappable(
        onPress: () async {
          await Clipboard.setData(ClipboardData(text: reaction.url));
          await incrementUsage(reaction.id);
          if (context.mounted) {
            showFToast(context: context, title: const Text('Link copied'));
          }
        },
        child: FCard.raw(
          clipBehavior: Clip.antiAlias,
          child: Stack(
            children: [
              SignalBuilder(
                builder: (context) {
                  final capturedDims = reactionDimensions.value[reaction.id];
                  final storedDims =
                      (reaction.width != null && reaction.height != null && reaction.height! > 0)
                          ? (reaction.width!, reaction.height!)
                          : null;
                  final effectiveDims = capturedDims;
                  final aspectRatio = effectiveDims != null && effectiveDims.$2 > 0
                      ? effectiveDims.$1 / effectiveDims.$2
                      : (storedDims != null && storedDims.$2 > 0
                          ? storedDims.$1 / storedDims.$2
                          : 1.0);

                  return AspectRatio(
                    aspectRatio: aspectRatio,
                    child: Image(
                      image: ReactionImageProvider(
                        reactionId: reaction.id,
                        url: reaction.url,
                        thumbnailStaticPath: thumbnailStaticAbs,
                        thumbnailAnimatedPath: thumbnailAnimatedAbs,
                        localFilePath: localFileAbs,
                        animatedPreviewsEnabled:
                            coordinator?.config.animatedPreviewsEnabled ?? false,
                        onDimensions: (w, h) {
                          final current = reactionDimensions.value[reaction.id];
                          if (current == null) {
                            final newDims = Map<String, (int, int)>.from(
                                reactionDimensions.value);
                            newDims[reaction.id] = (w, h);
                            reactionDimensions.value = newDims;
                            coordinator?.updateDimensions(reaction.id, w, h);
                          }
                        },
                      ),
                      fit: BoxFit.cover,
                      errorBuilder: (context, error, stack) => Container(
                        color: colors.muted,
                        child: Icon(FLucideIcons.imageOff,
                            size: 28, color: colors.mutedForeground),
                      ),
                      loadingBuilder: (context, child, loadingProgress) {
                        if (loadingProgress == null) return child;
                        return Container(
                          color: colors.muted,
                          child: Center(
                            child: SizedBox(
                              width: 24,
                              height: 24,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                value: loadingProgress.expectedTotalBytes != null
                                    ? loadingProgress.cumulativeBytesLoaded /
                                        loadingProgress.expectedTotalBytes!
                                    : null,
                              ),
                            ),
                          ),
                        );
                      },
                    ),
                  );
                },
              ),
              SignalBuilder(
                builder: (context) {
                  final status = ingestStatus.value[reaction.id];
                  if (status == null || status.isEmpty) {
                    return const SizedBox.shrink();
                  }
                  return Positioned(
                    left: 0,
                    right: 0,
                    bottom: 0,
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          begin: Alignment.topCenter,
                          end: Alignment.bottomCenter,
                          colors: [
                            Colors.transparent,
                            Colors.black.withValues(alpha: 0.7),
                          ],
                        ),
                      ),
                      child: Text(
                        status,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 11,
                          decoration: null,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  );
                },
              ),
              if (reaction.usageCount > 0)
                Positioned(
                  right: 4,
                  top: 4,
                  child: FBadge(
                    variant: FBadgeVariant.secondary,
                    child: Text('${reaction.usageCount}'),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
