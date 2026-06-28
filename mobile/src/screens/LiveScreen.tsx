import React, {useEffect, useRef, useState} from 'react';
import {
  ActivityIndicator,
  Image,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import type {NativeStackScreenProps} from '@react-navigation/native-stack';

import type {RootStackParamList} from '../../App';
import {FrameResult} from '../models/frameResult';
import {StreamService} from '../services/streamService';
import DetectionOverlay from '../components/DetectionOverlay';
import EventsList from '../components/EventsList';
import QodIndicator from '../components/QodIndicator';
import RiskBadge from '../components/RiskBadge';
import VehicleCard from '../components/VehicleCard';

/**
 * Canli ekran — backend WS akisindan gelen kareleri + tespitleri gosterir.
 * Sartname §3: tespit edilen veriler mobil uygulama ekraninda gosterilir.
 */
type Props = NativeStackScreenProps<RootStackParamList, 'Live'>;

export default function LiveScreen({route}: Props): React.JSX.Element {
  const {phone, source = '0'} = route.params;
  const [frame, setFrame] = useState<FrameResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [videoW, setVideoW] = useState(0);
  const serviceRef = useRef<StreamService | null>(null);

  useEffect(() => {
    const service = new StreamService();
    serviceRef.current = service;
    service.connect(
      {
        onFrame: f => {
          setError(null);
          setFrame(f);
        },
        onError: msg => setError(msg),
      },
      source,
      phone,
    );
    return () => service.close();
  }, [phone, source]);

  if (error != null && frame == null) {
    return (
      <View style={styles.center}>
        <Text style={styles.errorText}>
          Bağlantı hatası: {error}
          {'\n'}Backend (server.py) çalışıyor ve config.ts adresi doğru mu?
        </Text>
      </View>
    );
  }

  if (frame == null) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#3F51B5" />
      </View>
    );
  }

  const ratio =
    frame.width > 0 && frame.height > 0 ? frame.width / frame.height : 16 / 9;
  const videoH = videoW > 0 ? videoW / ratio : 0;

  return (
    <ScrollView style={styles.scroll}>
      <View
        style={[styles.videoArea, {height: videoH}]}
        onLayout={e => setVideoW(e.nativeEvent.layout.width)}>
        {frame.imageB64 ? (
          <Image
            style={StyleSheet.absoluteFill}
            source={{uri: `data:image/jpeg;base64,${frame.imageB64}`}}
            resizeMode="stretch"
          />
        ) : (
          <View style={[StyleSheet.absoluteFill, styles.blackFill]} />
        )}
        {videoW > 0 && videoH > 0 && (
          <DetectionOverlay
            detections={frame.detections}
            srcWidth={frame.width}
            srcHeight={frame.height}
            dispWidth={videoW}
            dispHeight={videoH}
          />
        )}
      </View>

      <View style={styles.badgeRow}>
        <RiskBadge risk={frame.risk} />
        <QodIndicator active={frame.qodActive} risk={frame.risk} />
      </View>

      <VehicleCard info={frame.vehicle} />

      <Text style={styles.sectionTitle}>Tespitler</Text>
      <EventsList events={frame.events} />
      <View style={{height: 16}} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: {flex: 1, backgroundColor: '#fff'},
  center: {flex: 1, justifyContent: 'center', alignItems: 'center', padding: 24},
  errorText: {textAlign: 'center', color: '#333'},
  videoArea: {width: '100%', backgroundColor: '#000', position: 'relative'},
  blackFill: {backgroundColor: '#000'},
  badgeRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    paddingHorizontal: 8,
    paddingVertical: 8,
  },
  sectionTitle: {
    fontWeight: 'bold',
    fontSize: 16,
    paddingLeft: 16,
    paddingTop: 8,
  },
});
