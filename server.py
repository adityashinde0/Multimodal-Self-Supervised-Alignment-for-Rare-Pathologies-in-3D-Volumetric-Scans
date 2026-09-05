"""PS-007 Multimodal Self-Supervised Alignment - Web Application Server
Serves the light-box frontend and provides live Zero-Shot Pathology Retrieval API.
"""

import os
import sys
import json
import time
import mimetypes
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import urllib.parse

# Setup path and suppress third-party warning noise
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
PORT = 8000

# Global retrieval engine reference
engine = None
dataset = None


def initialize_engine():
    global engine, dataset
    try:
        import torch
        from src.data.dataset import VolumeReportDataset
        from src.models.aligner import Multimodal3DAligner
        from src.eval.retrieval import ZeroShotRetrievalEngine

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[PS-007 Server] Loading 3D Dataset and initializing PyTorch engine on {device}...")

        dataset = VolumeReportDataset(data_dir=".", report_file="radiology_reports.json", augment=False)
        model = Multimodal3DAligner(
            img_size=(16, 16, 16),
            patch_size=(4, 4, 4),
            embed_dim=128,
            shared_dim=128,
            text_model_name="sentence-transformers/all-MiniLM-L6-v2",
            mask_ratio=0.75,
        ).to(device)

        ckpt_path = os.path.join("artifacts", "checkpoints", "multimodal_aligner.pt")
        if os.path.exists(ckpt_path):
            print(f"[PS-007 Server] Loading trained checkpoint: {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            print("[PS-007 Server] Checkpoint not found; using zero-shot representation baseline.")

        engine = ZeroShotRetrievalEngine(model, device=device)
        engine.index_gallery(dataset)
        print(f"[PS-007 Server] Successfully indexed {len(dataset)} 3D volumetric scans.")
        return True
    except Exception as e:
        print(f"[PS-007 Server] Warning: Could not initialize live PyTorch engine ({e}).")
        print("[PS-007 Server] Frontend will operate seamlessly in standalone mode.")
        return False


class RadiologyAppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            payload = {
                "status": "ok",
                "service": "PS-007 Multimodal Alignment Engine",
                "engine_loaded": engine is not None,
            }
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return

        # Default static file handling from frontend/
        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/query":
            content_length = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_length).decode("utf-8")

            try:
                data = json.loads(post_body)
                query_text = data.get("query", "").strip()
            except Exception:
                query_text = ""

            start_t = time.perf_counter()
            results = []

            if engine is not None and query_text:
                try:
                    raw_results = engine.query(query_text, top_k=5)
                    for item in raw_results:
                        case_idx = int(item["case_id"].split("_")[1])
                        results.append({
                            "rank": item["rank"],
                            "case_id": item["case_id"],
                            "pathology": item["pathology"],
                            "similarity_score": float(item["similarity_score"]),
                            "report": item.get("clinical_report", ""),
                            "case_idx": case_idx,
                        })
                except Exception as e:
                    print(f"[API Error] Retrieval failed: {e}")

            latency_ms = (time.perf_counter() - start_t) * 1000.0

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            resp = {
                "status": "ok",
                "query": query_text,
                "latency_ms": latency_ms,
                "results": results,
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        self.send_error(404, "Endpoint not found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        # Clean logging
        sys.stderr.write(f"[HTTP] {self.address_string()} - {format % args}\n")


def run_server(port=PORT):
    initialize_engine()
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, RadiologyAppHandler)
    url = f"http://localhost:{port}"
    print("=" * 65)
    print(f"  PS-007 RADIOLOGY LIGHT-BOX WEB APPLICATION RUNNING")
    print(f"  Open in your browser: {url}")
    print("=" * 65)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[PS-007 Server] Shutting down server gracefully...")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
