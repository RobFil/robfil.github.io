import { BarcodeFormat, BrowserMultiFormatReader } from "@zxing/browser";
import { ArrowLeft } from "lucide-react";
import { useEffect, useRef, useState } from "react";

export type ScanMode = "purchase" | "consume";

interface ScannerPageProps {
  mode: ScanMode;
  onCancel: () => void;
  onDetected: (barcode: string) => Promise<void>;
}

const supportedFormats = [
  BarcodeFormat.EAN_13,
  BarcodeFormat.EAN_8,
  BarcodeFormat.UPC_A,
  BarcodeFormat.UPC_E,
  BarcodeFormat.CODE_128,
];

export function ScannerPage({ mode, onCancel, onDetected }: ScannerPageProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const detectedRef = useRef(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stopped = false;
    let controls: { stop: () => void } | undefined;
    const reader = new BrowserMultiFormatReader();
    reader.possibleFormats = supportedFormats;

    async function start() {
      if (!videoRef.current) return;
      try {
        controls = await reader.decodeFromConstraints(
          { video: { facingMode: { ideal: "environment" } }, audio: false },
          videoRef.current,
          async (result) => {
            if (!result || detectedRef.current) return;
            detectedRef.current = true;
            controls?.stop();
            navigator.vibrate?.(45);
            await onDetected(result.getText());
          },
        );
        if (stopped) controls.stop();
      } catch (scanError) {
        if (!stopped) {
          const message = scanError instanceof DOMException && scanError.name === "NotAllowedError"
            ? "Die Kamera wurde nicht freigegeben. Erlaube den Kamerazugriff in Safari und versuche es erneut."
            : "Die Kamera konnte nicht gestartet werden.";
          setError(message);
        }
      }
    }

    void start();
    return () => {
      stopped = true;
      controls?.stop();
    };
  }, [onDetected]);

  return (
    <section className="page scanner-page">
      <header className="form-header">
        <button aria-label="Scanner schliessen" className="icon-button" onClick={onCancel} type="button"><ArrowLeft aria-hidden="true" size={22} /></button>
        <h1>{mode === "purchase" ? "Gekauft" : "Verbraucht"}</h1>
      </header>
      <div className="camera-frame">
        <video autoPlay className="camera-preview" muted playsInline ref={videoRef} />
        <div aria-hidden="true" className="scan-guide" />
      </div>
      <p className="scanner-copy">Barcode innerhalb des Rahmens halten.</p>
      {error && <p className="scanner-error" role="alert">{error}</p>}
    </section>
  );
}
