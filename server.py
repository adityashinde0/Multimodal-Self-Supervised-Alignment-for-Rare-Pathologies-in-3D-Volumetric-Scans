"""PS-007 Multimodal Self-Supervised Alignment - Robust Web Application Server
Serves the light-box frontend and provides live Zero-Shot Pathology Retrieval API.
"""

import os
import sys
import json
import time
import urllib.parse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

# Setup path and suppress third-party warning noise
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
PORT = 8000

# Global engine and operational metadata references
engine = None
dataset = None
engine_metadata = {
    "status": "uninitialized",
    "engine_loaded": False,
    "checkpoint_loaded": False,
    "checkpoint_path": None,
    "checkpoint_error": None,
    "device": "cpu",
    "cuda_available": False,
    "text_encoder": "uninitialized",
    "num_indexed_cases": 0,
    "mask_ratio": 0.75
}


def initialize_engine():
    global engine, dataset, engine_metadata
    try:
        import torch
        from src.data.dataset import VolumeReportDataset
        from src.models.aligner import Multimodal3DAligner
        from src.eval.retrieval import ZeroShotRetrievalEngine

        device = "cuda" if torch.cuda.is_available() else "cpu"
        engine_metadata["device"] = device
        engine_metadata["cuda_available"] = torch.cuda.is_available()
        print(f"[PS-007 Server] Loading 3D Dataset and initializing PyTorch engine on {device.upper()}...")

        dataset = VolumeReportDataset(data_dir=".", report_file="radiology_reports.json", augment=False)
        model = Multimodal3DAligner(
            img_size=(16, 16, 16),
            patch_size=(4, 4, 4),
            embed_dim=128,
            shared_dim=128,
            text_model_name="sentence-transformers/all-MiniLM-L6-v2",
            mask_ratio=0.75,
        ).to(device)
        model.eval()

        ckpt_path = os.path.join("artifacts", "checkpoints", "multimodal_aligner.pt")
        engine_metadata["checkpoint_path"] = ckpt_path

        if os.path.exists(ckpt_path):
            try:
                state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if len(missing) == 0:
                    print(f"[PS-007 Server] Trained checkpoint verified and loaded successfully: {ckpt_path}")
                    engine_metadata["checkpoint_loaded"] = True
                else:
                    print(f"[PS-007 Server] Checkpoint loaded with {len(missing)} unkeyed parameters: {missing}")
                    engine_metadata["checkpoint_loaded"] = True
            except Exception as ckpt_err:
                print(f"[PS-007 Server] ERROR: Corrupted or incompatible checkpoint ({ckpt_err})!")
                engine_metadata["checkpoint_loaded"] = False
                engine_metadata["checkpoint_error"] = str(ckpt_err)
        else:
            print(f"[PS-007 Server] Notice: No checkpoint found at {ckpt_path}. Operating with unaligned weights.")
            engine_metadata["checkpoint_loaded"] = False

        engine_metadata["text_encoder"] = model.active_encoder_name

        engine = ZeroShotRetrievalEngine(model, device=device)
        num_indexed = engine.index_gallery(dataset)
        engine_metadata["num_indexed_cases"] = num_indexed
        engine_metadata["engine_loaded"] = True
        engine_metadata["status"] = "ready"
        print(f"[PS-007 Server] Successfully indexed {num_indexed} 3D volumetric scans. Engine status: READY.")
        return True
    except Exception as e:
        print(f"[PS-007 Server] Warning: Could not initialize live PyTorch engine ({e}).")
        engine_metadata["status"] = "failed"
        engine_metadata["checkpoint_error"] = str(e)
        return False


class RadiologyAppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            payload = {
                "status": "ok",
                "service": "PS-007 Multimodal Alignment Engine",
                "metadata": engine_metadata
            }
            self.wfile.write(json.dumps(payload, indent=2).encode("utf-8"))
            return

        # Default static file handling from frontend/
        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/query":
            # 1. Verify engine availability
            if engine is None or not engine_metadata["engine_loaded"]:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                err_resp = {
                    "status": "error",
                    "error_code": "SERVICE_UNAVAILABLE",
                    "message": "PyTorch Zero-Shot Retrieval Engine is not initialized.",
                    "details": engine_metadata.get("checkpoint_error")
                }
                self.wfile.write(json.dumps(err_resp).encode("utf-8"))
                return

            # 2. Parse request payload
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "error",
                    "error_code": "EMPTY_PAYLOAD",
                    "message": "Request body must contain JSON with 'query' field."
                }).encode("utf-8"))
                return

            post_body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(post_body)
            except Exception as json_err:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "error",
                    "error_code": "INVALID_JSON",
                    "message": f"Malformed JSON request: {str(json_err)}"
                }).encode("utf-8"))
                return

            query_text = data.get("query", "").strip()
            top_k = data.get("top_k", 5)

            # 3. Validate query text
            if not query_text:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "error",
                    "error_code": "EMPTY_QUERY",
                    "message": "Clinical query text cannot be empty."
                }).encode("utf-8"))
                return

            # 4. Execute retrieval and measure exact latency
            start_t = time.perf_counter()
            try:
                raw_results = engine.query(query_text, top_k=top_k)
                latency_ms = (time.perf_counter() - start_t) * 1000.0

                results = []
                for item in raw_results:
                    case_idx = int(item["case_id"].split("_")[1])
                    results.append({
                        "rank": item["rank"],
                        "case_id": item["case_id"],
                        "pathology": item["pathology"],
                        "similarity_score": round(float(item["similarity_score"]), 4),
                        "report": item.get("clinical_report", ""),
                        "case_idx": case_idx,
                    })

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                resp = {
                    "status": "ok",
                    "query": query_text,
                    "latency_ms": round(latency_ms, 2),
                    "model_inference": "live_pytorch",
                    "checkpoint_loaded": engine_metadata["checkpoint_loaded"],
                    "device": engine_metadata["device"],
                    "results": results,
                }
                self.wfile.write(json.dumps(resp).encode("utf-8"))
                return

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                err_resp = {
                    "status": "error",
                    "error_code": "INFERENCE_ERROR",
                    "message": f"Inference execution failed: {str(e)}"
                }
                self.wfile.write(json.dumps(err_resp).encode("utf-8"))
                return

        self.send_error(404, "Endpoint not found")

    def log_message(self, format, *args):
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
