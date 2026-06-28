/**
 * Uygulama yapilandirmasi. backendHost'u, server.py'nin calistigi bilgisayarin
 * (telefonla AYNI ag / 5G) IP adresine gore DUZENLEYIN.
 */
export const AppConfig = {
  backendHost: '192.168.1.100', // <-- DUZENLE
  backendPort: 8000,

  get httpBase(): string {
    return `http://${this.backendHost}:${this.backendPort}`;
  },

  /** WS akis URL'i. source: video yolu / rtsp / kamera index (varsayilan "0"). */
  wsStream(source: string = '0', phone?: string): string {
    let url = `ws://${this.backendHost}:${this.backendPort}/ws/stream?source=${encodeURIComponent(
      source,
    )}`;
    if (phone && phone.length > 0) {
      url += `&phone=${encodeURIComponent(phone)}`;
    }
    return url;
  },
};
