import json
import time
import os
import shutil
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

from .settings import settings
from .state import SharedState

# ============================================================ 
# Dashboard HTML
# ============================================================ 
# (Ideally, this would be a separate file read at runtime, 
#  but for now we keep the structure similar to before or read from file if present)
#  The previous code read from 'web/index.html' if available, or fell back. 
#  The original main.py contained the HTML string as a fallback. 
#  I will read from file in the handler, consistent with the original logic.

class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler with Dashboard, MJPEG stream, and API endpoints."""
    
    def __init__(self, request, client_address, server, shared_state: SharedState):
        self.shared_state = shared_state
        super().__init__(request, client_address, server)
    
    def log_message(self, format, *args):
        pass  # Suppress logging
    
    def do_GET(self):
        if self.path in ('/', '/dashboard'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html_path = os.path.join(settings.PROJECT_ROOT, 'web', 'index.html')
            try:
                with open(html_path, 'r', encoding='utf-8') as f:
                    self.wfile.write(f.read().encode())
            except FileNotFoundError:
                self.wfile.write(b"<h1>Dashboard not found. Check web/index.html</h1>")
        
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Connection', 'close')
            self.end_headers()
            
            self.connection.settimeout(5.0)
            
            last_frame_id = -1
            skip_count = 0
            max_skip = 60
            
            while self.shared_state.is_running():
                jpeg_buffer, frame_id = self.shared_state.get_jpeg_buffer()
                
                if jpeg_buffer is None:
                    time.sleep(0.016)
                    continue
                
                if frame_id == last_frame_id:
                    skip_count += 1
                    if skip_count < max_skip:
                        time.sleep(0.016)
                        continue
                
                last_frame_id = frame_id
                skip_count = 0
                
                try:
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(jpeg_buffer)}\r\n\r\n'.encode())
                    self.wfile.write(jpeg_buffer)
                    self.wfile.write(b'\r\n')
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                    break
                
                time.sleep(0.033)
        
        elif self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            status = self.shared_state.get_api_status()
            self.wfile.write(json.dumps(status).encode())
        
        elif self.path.startswith('/api/events'):
            self._handle_events_api()
        
        elif self.path.startswith('/api/zones'):
            self._handle_zones_api()
        
        elif self.path == '/api/health':
            self._handle_health_api()
        
        elif self.path.startswith('/api/reports/trends'):
            self._handle_trends_api()
        
        else:
            self.send_error(404)
    
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else b'{}'
        
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}
        
        if self.path.startswith('/api/events/') and self.path.endswith('/ack'):
            try:
                event_id = int(self.path.split('/')[-2])
                if self.shared_state.event_store:
                    result = self.shared_state.event_store.acknowledge(event_id)
                    self._json_response({"success": result, "event_id": event_id})
                else:
                    self._json_response({"error": "Safety system not initialized"}, 500)
            except (ValueError, IndexError):
                self._json_response({"error": "Invalid event ID"}, 400)
        
        elif self.path.startswith('/api/events/') and self.path.endswith('/confirm'):
            try:
                event_id = int(self.path.split('/')[-2])
                if self.shared_state.event_store:
                    is_true = data.get('confirmed', True)
                    reason = data.get('reason', '')
                    result = self.shared_state.event_store.confirm(event_id, is_true, reason)
                    self._json_response({"success": result, "event_id": event_id})
                else:
                    self._json_response({"error": "Safety system not initialized"}, 500)
            except (ValueError, IndexError):
                self._json_response({"error": "Invalid event ID"}, 400)
        
        elif self.path == '/api/zones/silence':
            zone_id = data.get('zone_id')
            minutes = data.get('minutes', 5)
            
            if not zone_id:
                self._json_response({"error": "zone_id required"}, 400)
                return
            
            if self.shared_state.alert_manager:
                end_time = self.shared_state.alert_manager.silence_zone(zone_id, minutes)
                self._json_response({
                    "success": True,
                    "zone_id": zone_id,
                    "silenced_until": end_time.isoformat()
                })
            else:
                self._json_response({"error": "Safety system not initialized"}, 500)
        
        elif self.path == '/api/zones/unsilence':
            zone_id = data.get('zone_id')
            if not zone_id:
                self._json_response({"error": "zone_id required"}, 400)
                return
            
            if self.shared_state.alert_manager:
                self.shared_state.alert_manager.unsilence_zone(zone_id)
                self._json_response({"success": True, "zone_id": zone_id})
            else:
                self._json_response({"error": "Safety system not initialized"}, 500)
        else:
            self.send_error(404)
    
    def _json_response(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _handle_events_api(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        
        if not self.shared_state.event_store:
            self._json_response({"error": "Safety system not initialized"}, 500)
            return
        
        start = params.get('start', [None])[0]
        end = params.get('end', [None])[0]
        event_type = params.get('type', [None])[0]
        severity = params.get('severity', [None])[0]
        zone_id = params.get('zone_id', [None])[0]
        limit = int(params.get('limit', [50])[0])
        offset = int(params.get('offset', [0])[0])
        
        events = self.shared_state.event_store.query_events(
            start=start, end=end, event_type=event_type,
            severity=severity, zone_id=zone_id,
            limit=limit, offset=offset
        )
        
        self._json_response({
            "events": [e.to_dict() for e in events],
            "count": len(events),
            "limit": limit,
            "offset": offset
        })
    
    def _handle_zones_api(self):
        if not self.shared_state.zone_manager:
            self._json_response({"error": "Safety system not initialized"}, 500)
            return
        
        zones = []
        for zone in self.shared_state.zone_manager.get_all_zones():
            zones.append({
                "id": zone.id,
                "type": zone.zone_type.value,
                "display_name": zone.display_name or zone.id,
                "enabled": zone.enabled,
                "polygon": zone.polygon,
                "rules": [
                    {"type": r.rule_type.value, 
                     "max_dwell_seconds": r.max_dwell_seconds, 
                     "severity": r.severity.value} for r in zone.rules
                ]
            })
        
        silenced = {}
        if self.shared_state.alert_manager:
            silenced = self.shared_state.alert_manager.get_silenced_zones()
        
        self._json_response({"zones": zones, "silenced_zones": silenced})
    
    def _handle_health_api(self):
        disk = shutil.disk_usage(settings.PROJECT_ROOT)
        disk_free_gb = disk.free / (1024 ** 3)
        
        health = {
            "status": "healthy",
            "disk_free_gb": round(disk_free_gb, 2),
            "safety_enabled": self.shared_state.safety_enabled,
            "vlm_processing": self.shared_state.vlm_processing,
        }
        
        if self.shared_state.metrics:
            summary = self.shared_state.metrics.get_summary()
            health["fps"] = summary["fast_loop"]["fps"]
            health["uptime_s"] = summary["uptime_s"]
        
        if self.shared_state.zone_manager:
            health["zones_loaded"] = len(self.shared_state.zone_manager.zones)
        
        if self.shared_state.event_store:
            health["unacknowledged_events"] = self.shared_state.event_store.get_unacknowledged_count()
            health["today_summary"] = self.shared_state.event_store.get_daily_summary()
        
        self._json_response(health)
    
    def _handle_trends_api(self):
        """Return trend data for Chart.js visualization."""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        
        days = int(params.get('days', [7])[0])
        
        if not self.shared_state.event_store:
            self._json_response({"error": "Safety system not initialized"}, 500)
            return
        
        trends = self.shared_state.event_store.get_trends(days=days)
        self._json_response(trends)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

def run_web_server(shared_state: SharedState):
    """Run Dashboard web server."""
    # We need to partially apply the shared_state to the handler
    def handler_factory(*args, **kwargs):
        return DashboardHandler(*args, **kwargs, shared_state=shared_state)
    
    server = ThreadedHTTPServer(('0.0.0.0', settings.WEB_PORT), handler_factory)
    print(f"🌐 Dashboard at http://0.0.0.0:{settings.WEB_PORT}/dashboard")
    server.serve_forever()
