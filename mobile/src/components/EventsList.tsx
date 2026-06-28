import React from 'react';
import {StyleSheet, Text, View} from 'react-native';

import {DriverEvent} from '../models/frameResult';

/** Surucu eylemleri / tespit listesi (kategori + etiket + zaman + guven). */
interface Props {
  events: DriverEvent[];
}

function iconFor(etiket: string): string {
  switch (etiket) {
    case 'telefonla_konusma':
      return '📞';
    case 'sigara_icme':
      return '🚬';
    case 'su_icme':
      return '🥤';
    case 'esneme':
      return '🥱';
    case 'emniyet_kemeri_ihlali':
      return '💺';
    case 'slalom':
      return '〰️';
    case 'arkaya_bakma':
    case 'etrafa_bakinma':
      return '👀';
    default:
      return '⚠️';
  }
}

export default function EventsList({events}: Props): React.JSX.Element {
  if (events.length === 0) {
    return (
      <View style={styles.empty}>
        <Text style={styles.emptyText}>Tespit yok</Text>
      </View>
    );
  }
  return (
    <View>
      {events.map((e, i) => (
        <View key={i} style={styles.row}>
          <Text style={styles.icon}>{iconFor(e.etiket)}</Text>
          <View style={styles.body}>
            <Text style={styles.title}>{e.etiket}</Text>
            <Text style={styles.subtitle}>
              {e.kategori} • {e.zamanSaniye.toFixed(1)} sn
            </Text>
          </View>
          <Text style={styles.conf}>
            {Math.round(e.confidenceScore * 100)}%
          </Text>
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  empty: {padding: 12},
  emptyText: {color: '#999'},
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  icon: {fontSize: 20, width: 30},
  body: {flex: 1},
  title: {fontSize: 15, color: '#111'},
  subtitle: {fontSize: 12, color: '#777'},
  conf: {fontSize: 13, color: '#444', fontWeight: '600'},
});
