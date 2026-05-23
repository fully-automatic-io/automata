use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

static LAST_TIMESTAMP: Mutex<u64> = Mutex::new(0);
static SEQUENCE: AtomicU32 = AtomicU32::new(0);

pub fn uuidv7() -> String {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    let (ts, seq) = {
        let mut last = LAST_TIMESTAMP.lock().unwrap();
        if now_ms > *last {
            *last = now_ms;
            let s = rand_u32() & 0x0FFF_FFFF;
            SEQUENCE.store(s, Ordering::SeqCst);
            (now_ms, s)
        } else {
            let s = SEQUENCE.fetch_add(1, Ordering::SeqCst).wrapping_add(1);
            if s == 0 {
                *last += 1;
            }
            (*last, s)
        }
    };

    let mut b = [0u8; 16];
    b[0] = (ts >> 40) as u8;
    b[1] = (ts >> 32) as u8;
    b[2] = (ts >> 24) as u8;
    b[3] = (ts >> 16) as u8;
    b[4] = (ts >> 8) as u8;
    b[5] = ts as u8;
    b[6] = 0x70 | ((seq >> 24) & 0x0F) as u8;
    b[7] = (seq >> 16) as u8;
    b[8] = 0x80 | ((seq >> 10) & 0x3F) as u8;
    b[9] = (seq >> 2) as u8;
    let r = rand_bytes();
    b[10] = ((seq & 0x03) << 6) as u8 | (r[0] & 0x3F);
    b[11..16].copy_from_slice(&r[1..6]);

    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],b[8],b[9],b[10],b[11],b[12],b[13],b[14],b[15]
    )
}

fn rand_u32() -> u32 {
    let r = rand_bytes();
    u32::from_le_bytes([r[0], r[1], r[2], r[3]])
}

fn rand_bytes() -> [u8; 6] {
    let id = uuid::Uuid::new_v4();
    let b = id.as_bytes();
    [b[0], b[1], b[2], b[3], b[4], b[5]]
}

pub fn short_id() -> String {
    uuidv7()[..8].to_string()
}

pub fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339()
}
