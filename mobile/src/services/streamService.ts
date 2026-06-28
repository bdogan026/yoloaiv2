import {AppConfig} from '../config';
import {FrameResult, tryParseFrame} from '../models/frameResult';

export interface StreamCallbacks {
  onFrame: (frame: FrameResult) => void;
  onError?: (message: string) => void;
  onClose?: () => void;
}

/**
 * WebSocket uzerinden canli tespit akisi. server.py /ws/stream'e baglanir,
 * gelen her JSON mesajini FrameResult'a cevirip onFrame ile iletir.
 * React Native global WebSocket'i kullanir (ek bagimlilik yok).
 */
export class StreamService {
  private ws: WebSocket | null = null;

  connect(callbacks: StreamCallbacks, source: string = '0', phone?: string): void {
    const url = AppConfig.wsStream(source, phone);
    const ws = new WebSocket(url);
    this.ws = ws;

    ws.onmessage = (event: WebSocketMessageEvent) => {
      const data = event.data;
      if (typeof data !== 'string') {
        return;
      }
      const frame = tryParseFrame(data);
      if (frame) {
        callbacks.onFrame(frame);
      }
    };

    ws.onerror = (event: any) => {
      callbacks.onError?.(event?.message ?? 'WebSocket hatasi');
    };

    ws.onclose = () => {
      callbacks.onClose?.();
    };
  }

  close(): void {
    this.ws?.close();
    this.ws = null;
  }
}
