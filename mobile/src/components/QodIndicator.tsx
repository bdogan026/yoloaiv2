import React from 'react';
import {StyleSheet, Text, View} from 'react-native';

import {RiskInfo} from '../models/frameResult';

/**
 * QoD durumu gostergesi: AKTIF (yuksek kalite talep edildi) / pasif + ihtiyac.
 * Sartname %40: QoD'un YALNIZCA ihtiyac varken acildigini gorsel kanitlar.
 */
interface Props {
  active: boolean;
  risk: RiskInfo;
}

export default function QodIndicator({active, risk}: Props): React.JSX.Element {
  const c = active ? '#2196F3' : '#9E9E9E';
  return (
    <View style={[styles.badge, {borderColor: c, backgroundColor: c + '26'}]}>
      <Text style={[styles.icon, {color: c}]}>{active ? '📶' : '📡'}</Text>
      <Text style={[styles.text, {color: c}]}>
        {active ? 'QoD AKTİF' : 'QoD pasif'}
      </Text>
      <Text style={[styles.need, {color: c}]}>
        ihtiyaç: {Math.round(risk.need * 100)}%
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderWidth: 1.5,
    borderRadius: 8,
  },
  icon: {fontSize: 14, marginRight: 6},
  text: {fontWeight: 'bold'},
  need: {fontSize: 12, marginLeft: 8},
});
