import React from 'react';
import Svg, {Rect, Text as SvgText} from 'react-native-svg';

import {Detection} from '../models/frameResult';

/**
 * Tespit kutularini video uzerine cizer. Kutular ORIJINAL piksel
 * koordinatlarindadir; gosterim boyutuna gore olceklenir.
 */
interface Props {
  detections: Detection[];
  srcWidth: number;
  srcHeight: number;
  dispWidth: number;
  dispHeight: number;
}

const VEHICLES = new Set([
  'sedan',
  'suv',
  'hatchback',
  'pickup',
  'minibus',
  'panelvan',
  'kamyon',
]);

function colorFor(cls: string): string {
  if (VEHICLES.has(cls)) {
    return '#69F0AE'; // greenAccent
  }
  if (cls === 'plaka') {
    return '#FFC107'; // amber
  }
  if (cls === 'kisi') {
    return '#40C4FF'; // lightBlueAccent
  }
  if (cls === 'teknocan' || cls === 'bilgisayar') {
    return '#FFAB40'; // orangeAccent
  }
  if (cls === 'telefon' || cls === 'sigara' || cls === 'sise') {
    return '#FF5252'; // redAccent
  }
  return '#FFFFFF';
}

export default function DetectionOverlay({
  detections,
  srcWidth,
  srcHeight,
  dispWidth,
  dispHeight,
}: Props): React.JSX.Element | null {
  if (srcWidth <= 0 || srcHeight <= 0) {
    return null;
  }
  const sx = dispWidth / srcWidth;
  const sy = dispHeight / srcHeight;

  return (
    <Svg
      style={{position: 'absolute', top: 0, left: 0}}
      width={dispWidth}
      height={dispHeight}>
      {detections.map((d, i) => {
        if (d.box.length < 4) {
          return null;
        }
        const color = colorFor(d.cls);
        const x = d.box[0] * sx;
        const y = d.box[1] * sy;
        const w = (d.box[2] - d.box[0]) * sx;
        const h = (d.box[3] - d.box[1]) * sy;
        const pct = Math.round(d.conf * 100);
        const label =
          d.trackId != null
            ? `${d.cls} #${d.trackId} ${pct}%`
            : `${d.cls} ${pct}%`;
        const labelY = Math.max(10, y - 4);
        return (
          <React.Fragment key={i}>
            <Rect
              x={x}
              y={y}
              width={w}
              height={h}
              stroke={color}
              strokeWidth={2}
              fill="none"
            />
            <SvgText
              x={x + 2}
              y={labelY}
              fill={color}
              fontSize={11}
              fontWeight="600">
              {label}
            </SvgText>
          </React.Fragment>
        );
      })}
    </Svg>
  );
}
