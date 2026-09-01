import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const html = readFileSync(new URL('./index.html', import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const match = html.match(/const markerSeriesHandler = \{[\s\S]*?\n\};\n\nfunction initChart/);
if (!match) throw new Error('index.html 中未找到 markerSeriesHandler');
const source = match[0].replace(/\n\nfunction initChart$/, '');

test('marker labels use the current time coordinate and ignore stale offscreen coordinates', () => {
  let capturedX = null;
  const context = {
    chart: { timeScale: () => ({ timeToCoordinate: () => 120 }) },
    markerState: null,
    drawArrow: () => {},
    trendLabel: () => '趋势1',
    ReversalMarkerLayout: {
      layoutReversalLabels: items => {
        capturedX = items[0]?.x ?? null;
        return items.map(item => ({ key: item.key, showLabel: true, labelY: item.baseY }));
      },
    },
  };
  vm.runInNewContext(`${source}\nglobalThis.markerSeriesHandler = markerSeriesHandler;`, context);
  context.markerSeriesHandler.update({
    bars: [{ originalData: { markerKey: '2026-08-28', ref: 10, high: 12, low: 8, up: 0, down: 0, reversal: true, trend: null, badge: false }, x: 999 }],
    barSpacing: 10,
  });
  const ctx = { measureText: () => ({ width: 20 }), fillText: () => {}, beginPath: () => {}, arc: () => {}, fill: () => {}, closePath: () => {} };
  context.markerSeriesHandler.renderer().draw({ useMediaCoordinateSpace: callback => callback({ context: ctx, mediaSize: { width: 300, height: 200 } }) }, () => 100);
  assert.equal(capturedX, 120);
});
