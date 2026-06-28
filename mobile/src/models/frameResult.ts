/**
 * Backend'den (server.py /ws/stream) gelen kare bazli sonuc modelleri.
 * Sunucu her kare icin JSON yayinlar; alanlar Flutter istemcisiyle ayni semadir.
 */

export interface Detection {
  cls: string;
  conf: number;
  box: number[]; // x1,y1,x2,y2 (ORIJINAL piksel koordinatlari)
  trackId?: number | null;
}

export interface VehicleInfo {
  tip: string;
  plaka: string;
  renk: string;
  hizKmh?: number | null;
  confidenceScore: number;
}

export interface DriverEvent {
  zamanSaniye: number;
  kategori: string;
  etiket: string;
  confidenceScore: number;
}

export interface RiskInfo {
  score: number;
  profile: string; // Guvenli | Dikkatli | Riskli | Tehlikeli
  need: number;
}

export interface FrameResult {
  frameIdx: number;
  width: number;
  height: number;
  vehicle: VehicleInfo;
  detections: Detection[];
  events: DriverEvent[];
  risk: RiskInfo;
  qodActive: boolean;
  imageB64?: string | null; // JPEG base64 ("data:" uri olmadan, ham base64)
}

const num = (v: unknown, def = 0): number =>
  typeof v === 'number' ? v : v == null ? def : Number(v) || def;

function parseDetection(j: any): Detection {
  return {
    cls: (j?.cls ?? '') as string,
    conf: num(j?.conf),
    box: Array.isArray(j?.box) ? j.box.map((e: unknown) => num(e)) : [],
    trackId: j?.track_id ?? null,
  };
}

function parseVehicle(j: any): VehicleInfo {
  return {
    tip: (j?.tip ?? '') as string,
    plaka: (j?.plaka ?? '') as string,
    renk: (j?.renk ?? '') as string,
    hizKmh: j?.hiz_kmh == null ? null : num(j.hiz_kmh),
    confidenceScore: num(j?.confidence_score),
  };
}

function parseEvent(j: any): DriverEvent {
  return {
    zamanSaniye: num(j?.zaman_saniye),
    kategori: (j?.kategori ?? '') as string,
    etiket: (j?.etiket ?? '') as string,
    confidenceScore: num(j?.confidence_score),
  };
}

function parseRisk(j: any): RiskInfo {
  return {
    score: num(j?.score),
    profile: (j?.profile ?? '-') as string,
    need: num(j?.need),
  };
}

export function frameFromJson(j: any): FrameResult {
  const qod = j?.qod ?? {};
  return {
    frameIdx: num(j?.frame_idx),
    width: num(j?.width, 1),
    height: num(j?.height, 1),
    vehicle: parseVehicle(j?.arac_bilgisi ?? {}),
    detections: Array.isArray(j?.detections) ? j.detections.map(parseDetection) : [],
    events: Array.isArray(j?.tespitler) ? j.tespitler.map(parseEvent) : [],
    risk: parseRisk(j?.risk ?? {}),
    qodActive: Boolean(qod?.active ?? false),
    imageB64:
      typeof j?.image_jpeg_b64 === 'string' && j.image_jpeg_b64.length > 0
        ? j.image_jpeg_b64
        : null,
  };
}

/** Ham WS string mesajini guvenli sekilde FrameResult'a cevirir (hata -> null). */
export function tryParseFrame(raw: string): FrameResult | null {
  try {
    return frameFromJson(JSON.parse(raw));
  } catch {
    return null;
  }
}
