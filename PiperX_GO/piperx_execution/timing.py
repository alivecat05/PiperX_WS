"""Four-timestamp clock calibration and bounded one-way latency estimates."""
import time


class ClockEstimate:
    def __init__(self):
        self.offset = None  # server monotonic - client monotonic
        self.uncertainty = None
        self.updated_at = None

    def update(self, client_send, server_receive, server_send, client_receive):
        delay = (client_receive-client_send)-(server_send-server_receive)
        if delay < -1e-6 or server_send < server_receive or client_receive < client_send:
            raise ValueError('Invalid clock calibration timestamps')
        delay = max(0.,delay)
        candidate = ((server_receive-client_send)+(server_send-client_receive))/2
        if self.uncertainty is None or delay/2 < self.uncertainty:
            self.offset = candidate
            self.uncertainty = delay/2
            self.updated_at = client_receive

    def estimate(self, client_send, server_receive, server_send, client_receive):
        if self.offset is None:
            raise RuntimeError('Clock calibration is required for one-way timing')
        age = max(0.,client_receive-self.updated_at)
        # Conservative allowance for relative clock drift between calibrations.
        error = self.uncertainty+age*200e-6
        upload = server_receive-client_send-self.offset
        download = client_receive-server_send+self.offset
        return dict(upload_ms_est=upload*1000,download_ms_est=download*1000,
            one_way_uncertainty_ms=error*1000,
            upload_bounds_ms=[max(0.,upload-error)*1000,max(0.,upload+error)*1000],
            download_bounds_ms=[max(0.,download-error)*1000,max(0.,download+error)*1000],
            network_roundtrip_ms=((client_receive-client_send)-(server_send-server_receive))*1000,
            clock_sample_age_s=age,timing_method='four_timestamp_min_rtt_200ppm_drift_estimate')
