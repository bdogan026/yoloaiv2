/**
 * @format
 * React Native giris noktasi. App'i app.json'daki isimle kaydeder.
 */
import {AppRegistry} from 'react-native';
import App from './App';
import {name as appName} from './app.json';

AppRegistry.registerComponent(appName, () => App);
