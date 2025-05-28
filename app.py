import cv2
from ultralytics import YOLO
import numpy as np

class FaceDetectionApp:
    def __init__(self, model_path="best.pt"):
        """
        Inisialisasi aplikasi deteksi wajah
        Args:
            model_path: Path ke model YOLO yang sudah dilatih
        """
        try:
            self.model = YOLO(model_path)
            print(f"Model YOLO berhasil dimuat dari: {model_path}")
        except Exception as e:
            print(f"Error memuat model: {e}")
            return
        
        # Inisialisasi kamera
        self.cap = cv2.VideoCapture(0)
        
        # Cek apakah kamera berhasil terbuka
        if not self.cap.isOpened():
            print("Error: Tidak dapat membuka kamera")
            return
        
        # Set resolusi kamera (optional)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Aplikasi deteksi wajah siap!")
        print("Tekan 'q' untuk keluar")
    
    def draw_detections(self, frame, results):
        """
        Menggambar bounding box dan label pada frame
        """
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Ambil koordinat bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Ambil confidence score
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Ambil class name (jika ada)
                    if hasattr(box, 'cls') and box.cls is not None:
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names.get(class_id, 'Face')
                    else:
                        class_name = 'Face'
                    
                    # Gambar bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Buat label dengan confidence
                    label = f"{class_name}: {confidence:.2f}"
                    
                    # Gambar background untuk text
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    
                    # Gambar text
                    cv2.putText(frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def add_info_overlay(self, frame):
        """
        Menambahkan informasi overlay pada frame
        """
        # Tambahkan judul
        cv2.putText(frame, "Face Detection - YOLO", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Tambahkan instruksi
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """
        Menjalankan aplikasi deteksi wajah
        """
        if not hasattr(self, 'model') or not hasattr(self, 'cap'):
            print("Aplikasi tidak dapat dijalankan karena error inisialisasi")
            return
        
        frame_count = 0
        
        while True:
            # Baca frame dari kamera
            ret, frame = self.cap.read()
            
            if not ret:
                print("Error: Tidak dapat membaca frame dari kamera")
                break
            
            # Flip frame secara horizontal untuk efek mirror
            frame = cv2.flip(frame, 1)
            
            # Lakukan deteksi setiap beberapa frame untuk performa yang lebih baik
            if frame_count % 2 == 0:  # Deteksi setiap 2 frame
                try:
                    # Jalankan deteksi YOLO
                    results = self.model(frame, verbose=False)
                    
                    # Gambar hasil deteksi
                    frame = self.draw_detections(frame, results)
                    
                    # Simpan results untuk frame berikutnya
                    self.last_results = results
                    
                except Exception as e:
                    print(f"Error saat deteksi: {e}")
                    # Gunakan hasil deteksi sebelumnya jika ada error
                    if hasattr(self, 'last_results'):
                        frame = self.draw_detections(frame, self.last_results)
            
            else:
                # Gunakan hasil deteksi sebelumnya untuk frame yang tidak dideteksi
                if hasattr(self, 'last_results'):
                    frame = self.draw_detections(frame, self.last_results)
            
            # Tambahkan overlay informasi
            frame = self.add_info_overlay(frame)
            
            # Tampilkan frame
            cv2.imshow('Face Detection - YOLO', frame)
            
            # Cek input keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                # Simpan screenshot
                filename = f"face_detection_screenshot_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot disimpan: {filename}")
            
            frame_count += 1
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Aplikasi ditutup")

def main():
    """
    Fungsi utama untuk menjalankan aplikasi
    """
    print("=== Aplikasi Deteksi Wajah Real-time ===")
    print("Menggunakan YOLO dan OpenCV")
    print()
    
    # Path ke model YOLO (sesuaikan dengan lokasi file model Anda)
    model_path = "best.pt"
    
    # Buat dan jalankan aplikasi
    app = FaceDetectionApp(model_path)
    app.run()

if __name__ == "__main__":
    main()