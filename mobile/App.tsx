/**
 * Rapid Response — 5G & YZ ile Akilli Yol Guvenligi mobil uygulamasi.
 * React Native 0.73 / TypeScript (OTR: "Mobil Uygulama (React Native 0.73 / TypeScript)").
 */
import React from 'react';
import {StatusBar} from 'react-native';
import {NavigationContainer} from '@react-navigation/native';
import {createNativeStackNavigator} from '@react-navigation/native-stack';

import LoginScreen from './src/screens/LoginScreen';
import LiveScreen from './src/screens/LiveScreen';

export type RootStackParamList = {
  Login: undefined;
  Live: {phone: string; source?: string};
};

const Stack = createNativeStackNavigator<RootStackParamList>();

const INDIGO = '#3F51B5';

export default function App(): React.JSX.Element {
  return (
    <NavigationContainer>
      <StatusBar barStyle="light-content" backgroundColor={INDIGO} />
      <Stack.Navigator
        screenOptions={{
          headerStyle: {backgroundColor: INDIGO},
          headerTintColor: '#fff',
          headerTitleStyle: {fontWeight: 'bold'},
        }}>
        <Stack.Screen
          name="Login"
          component={LoginScreen}
          options={{title: 'Rapid Response — Giriş'}}
        />
        <Stack.Screen
          name="Live"
          component={LiveScreen}
          options={{title: 'Canlı Analiz'}}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
