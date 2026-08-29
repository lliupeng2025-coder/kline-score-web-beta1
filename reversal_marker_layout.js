(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ReversalMarkerLayout = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  const intersects = (a, b) =>
    a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;

  function layoutReversalLabels(items, options) {
    const labelWidth = Math.max(1, Number(options && options.labelWidth) || 1);
    const labelHeight = Math.max(1, Number(options && options.labelHeight) || 1);
    const laneCount = Math.max(1, Math.floor(Number(options && options.laneCount) || 1));
    const laneGap = Math.max(0, Number(options && options.laneGap) || 0);
    const minHorizontalGap = Math.max(
      labelWidth + 8,
      Number(options && options.minHorizontalGap) || 0
    );
    const accepted = [];

    return (Array.isArray(items) ? items : [])
      .filter(item => item && Number.isFinite(item.x) && Number.isFinite(item.baseY))
      .sort((a, b) => a.x - b.x)
      .map(item => {
        if (accepted.some(other => Math.abs(item.x - other.x) < minHorizontalGap)) {
          return { key: item.key, showLabel: false, labelY: item.baseY, rect: null };
        }
        for (let lane = 0; lane < laneCount; lane++) {
          const labelY = item.baseY - lane * (labelHeight + laneGap);
          const rect = {
            left: item.x - labelWidth / 2,
            right: item.x + labelWidth / 2,
            top: labelY - labelHeight / 2,
            bottom: labelY + labelHeight / 2,
          };
          if (!accepted.some(other => intersects(rect, other.rect))) {
            accepted.push({ x: item.x, rect });
            return { key: item.key, showLabel: true, labelY, rect };
          }
        }
        return { key: item.key, showLabel: false, labelY: item.baseY, rect: null };
      });
  }

  return { layoutReversalLabels };
});
