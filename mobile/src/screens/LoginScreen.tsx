import React, {useState} from 'react';
import {
  ActivityIndicator,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import type {NativeStackScreenProps} from '@react-navigation/native-stack';

import type {RootStackParamList} from '../../App';
import {verifyNumber} from '../services/authService';

/**
 * Giris ekrani — Number Verification (sessiz, sebeke-tabanli) dogrulamasi.
 * Sartname §4.1: SMS/OTP yok; dogrulama mobil sebeke uzerinden arka planda.
 */
type Props = NativeStackScreenProps<RootStackParamList, 'Login'>;

export default function LoginScreen({navigation}: Props): React.JSX.Element {
  const [phone, setPhone] = useState('+90');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onVerify = async () => {
    setLoading(true);
    setError(null);
    const ok = await verifyNumber(phone.trim());
    setLoading(false);
    if (ok) {
      navigation.navigate('Live', {phone: phone.trim()});
    } else {
      setError(
        'Doğrulama başarısız. Şebeke/numara veya backend erişimini kontrol edin.',
      );
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.card}>
        <Text style={styles.icon}>🛡️</Text>
        <Text style={styles.subtitle}>
          Sessiz, şebeke-tabanlı doğrulama (Number Verification).{'\n'}
          SMS / kod girişi yok.
        </Text>

        <TextInput
          style={styles.input}
          value={phone}
          onChangeText={setPhone}
          keyboardType="phone-pad"
          placeholder="Telefon numarası"
          placeholderTextColor="#999"
          autoCapitalize="none"
        />

        {error != null && <Text style={styles.error}>{error}</Text>}

        <TouchableOpacity
          style={[styles.button, loading && styles.buttonDisabled]}
          onPress={onVerify}
          disabled={loading}
          activeOpacity={0.8}>
          {loading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.buttonText}>Doğrula &amp; Başla</Text>
          )}
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, justifyContent: 'center', backgroundColor: '#fff'},
  card: {alignSelf: 'center', maxWidth: 420, width: '100%', padding: 24},
  icon: {fontSize: 56, textAlign: 'center', marginBottom: 12},
  subtitle: {textAlign: 'center', color: '#777', marginBottom: 24},
  input: {
    borderWidth: 1,
    borderColor: '#bbb',
    borderRadius: 6,
    paddingHorizontal: 12,
    paddingVertical: 12,
    fontSize: 16,
    color: '#111',
    marginBottom: 16,
  },
  error: {color: '#d32f2f', marginBottom: 12},
  button: {
    backgroundColor: '#3F51B5',
    borderRadius: 6,
    paddingVertical: 14,
    alignItems: 'center',
  },
  buttonDisabled: {opacity: 0.6},
  buttonText: {color: '#fff', fontWeight: 'bold', fontSize: 16},
});
