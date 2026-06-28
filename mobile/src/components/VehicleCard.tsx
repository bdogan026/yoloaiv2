import React from 'react';
import {StyleSheet, Text, View} from 'react-native';

import {VehicleInfo} from '../models/frameResult';

/** Arac bilgisi karti: tip / plaka / renk / HIZ (Sartname §5) + guven. */
interface Props {
  info: VehicleInfo;
}

function KV({
  label,
  value,
  highlight = false,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}): React.JSX.Element {
  return (
    <View style={styles.kv}>
      <Text style={styles.kvKey}>{label}</Text>
      <Text
        style={[
          styles.kvVal,
          highlight && styles.kvValHighlight,
        ]}>
        {value}
      </Text>
    </View>
  );
}

export default function VehicleCard({info}: Props): React.JSX.Element {
  const hiz = info.hizKmh == null ? '—' : `${Math.round(info.hizKmh)} km/h`;
  return (
    <View style={styles.card}>
      <Text style={styles.title}>Araç Bilgisi</Text>
      <View style={styles.grid}>
        <KV label="Plaka" value={info.plaka || '—'} />
        <KV label="Tip" value={info.tip || '—'} />
        <KV label="Renk" value={info.renk || '—'} />
        <KV label="Hız" value={hiz} highlight />
        <KV
          label="Güven"
          value={`${Math.round(info.confidenceScore * 100)}%`}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    margin: 8,
    padding: 12,
    borderRadius: 8,
    backgroundColor: '#fff',
    elevation: 2,
    shadowColor: '#000',
    shadowOpacity: 0.1,
    shadowRadius: 4,
    shadowOffset: {width: 0, height: 1},
  },
  title: {fontWeight: 'bold', fontSize: 16, marginBottom: 8},
  grid: {flexDirection: 'row', flexWrap: 'wrap'},
  kv: {marginRight: 16, marginBottom: 8, minWidth: 70},
  kvKey: {fontSize: 12, color: '#999'},
  kvVal: {fontSize: 15, fontWeight: '500', color: '#222'},
  kvValHighlight: {fontSize: 18, fontWeight: 'bold', color: '#3F51B5'},
});
