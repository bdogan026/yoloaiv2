import React from 'react';
import {StyleSheet, Text, View} from 'react-native';

import {RiskInfo} from '../models/frameResult';

/** Risk profili rozeti (Guvenli/Dikkatli/Riskli/Tehlikeli) + skor. */
interface Props {
  risk: RiskInfo;
}

function colorFor(profile: string): string {
  switch (profile) {
    case 'Tehlikeli':
      return '#F44336';
    case 'Riskli':
      return '#FF5722';
    case 'Dikkatli':
      return '#FFA000';
    default:
      return '#4CAF50';
  }
}

export default function RiskBadge({risk}: Props): React.JSX.Element {
  const c = colorFor(risk.profile);
  return (
    <View style={[styles.badge, {borderColor: c, backgroundColor: c + '26'}]}>
      <Text style={[styles.icon, {color: c}]}>🛡️</Text>
      <Text style={[styles.text, {color: c}]}>
        {risk.profile} ({Math.round(risk.score)})
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
});
