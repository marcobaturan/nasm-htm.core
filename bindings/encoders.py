"""
nasm-htm.core Encoders Module
-----------------------------
Provides standard HTM encoders for converting real-world inputs (scalars, dates,
categories) into Sparse Distributed Representation (SDR) uint8 bit arrays.
"""

import numpy as np
import datetime

class ScalarEncoder:
    """
    Encodes continuous scalar numbers into an N-bit SDR with W active bits.
    """
    def __init__(self, n=2048, w=41, min_val=0.0, max_val=100.0, periodic=False):
        self.n = int(n)
        self.w = int(w)
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.periodic = periodic
        self.num_buckets = self.n - self.w + 1 if not periodic else self.n

    def encode(self, val):
        val = np.clip(val, self.min_val, self.max_val)
        norm_val = (val - self.min_val) / (self.max_val - self.min_val + 1e-12)
        bucket_idx = int(norm_val * (self.num_buckets - 1))
        
        sdr = np.zeros(self.n, dtype=np.uint8)
        if not self.periodic:
            sdr[bucket_idx : bucket_idx + self.w] = 1
        else:
            indices = [(bucket_idx + i) % self.n for i in range(self.w)]
            sdr[indices] = 1
        return sdr


class CategoryEncoder:
    """
    Encodes discrete categorical labels into orthogonal N-bit SDRs with W active bits.
    """
    def __init__(self, categories, n=2048, w=41):
        self.categories = list(categories)
        self.n = int(n)
        self.w = int(w)
        self.cat_to_sdr = {}
        
        for idx, cat in enumerate(self.categories):
            sdr = np.zeros(self.n, dtype=np.uint8)
            start_bit = (idx * self.w) % (self.n - self.w + 1)
            sdr[start_bit : start_bit + self.w] = 1
            self.cat_to_sdr[cat] = sdr

    def encode(self, category):
        if category in self.cat_to_sdr:
            return self.cat_to_sdr[category]
        sdr = np.zeros(self.n, dtype=np.uint8)
        sdr[:self.w] = 1
        return sdr


class DateEncoder:
    """
    Encodes date/time objects into concatenated SDR vectors (Day of Week + Time of Day).
    """
    def __init__(self, n_day=1024, w_day=21, n_time=1024, w_time=21):
        self.day_encoder = ScalarEncoder(n=n_day, w=w_day, min_val=0, max_val=6, periodic=True)
        self.time_encoder = ScalarEncoder(n=n_time, w=w_time, min_val=0, max_val=86400, periodic=True)
        self.n = n_day + n_time

    def encode(self, dt):
        if not isinstance(dt, datetime.datetime):
            dt = datetime.datetime.fromtimestamp(dt)
        
        day_sdr = self.day_encoder.encode(dt.weekday())
        seconds_in_day = dt.hour * 3600 + dt.minute * 60 + dt.second
        time_sdr = self.time_encoder.encode(seconds_in_day)
        
        return np.concatenate([day_sdr, time_sdr]).astype(np.uint8)
