"""
Supabase client for real-time navigation data streaming.
Handles connection, data insertion, and real-time updates.
"""

import os
import time
import logging
from typing import Dict, Optional, List
from datetime import datetime
from dotenv import load_dotenv

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("⚠ Supabase not installed. Install with: pip install supabase")

logger = logging.getLogger(__name__)


class SupabaseNavigationClient:
    """Real-time navigation data streaming to Supabase."""
    
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None,
                 table_name: Optional[str] = None):
        """
        Initialize Supabase client.
        
        Args:
            url: Supabase project URL (or from .env)
            key: Supabase API key (or from .env)
            table_name: Database table name (default: navigation_data)
        """
        if not SUPABASE_AVAILABLE:
            raise ImportError("supabase-py not installed")
        
        # Load environment variables
        load_dotenv()
        
        self.url = url or os.getenv('SUPABASE_URL')
        self.key = key or os.getenv('SUPABASE_KEY')
        self.table_name = table_name or os.getenv('SUPABASE_TABLE_NAME', 'navigation_data')
        
        if not self.url or not self.key:
            raise ValueError(
                "Supabase credentials not found. Set SUPABASE_URL and SUPABASE_KEY "
                "in .env file or pass them to constructor."
            )
        
        self.client: Optional[Client] = None
        self.connected = False
        self.session_id = None
        
    def connect(self):
        """Establish connection to Supabase."""
        try:
            self.client = create_client(self.url, self.key)
            
            # Generate unique session ID
            self.session_id = f"session_{int(time.time())}"
            
            self.connected = True
            logger.info(f"✓ Connected to Supabase (Session: {self.session_id})")
            
        except Exception as e:
            logger.error(f"✗ Failed to connect to Supabase: {e}")
            self.connected = False
            raise
    
    def disconnect(self):
        """Disconnect from Supabase."""
        if self.client:
            self.client = None
            self.connected = False
            logger.info("Disconnected from Supabase")
    
    def insert_navigation_data(self, nav_data: Dict) -> bool:
        """
        Insert navigation solution to database.
        
        Args:
            nav_data: Dictionary with navigation data
                Required fields: lat, lon, alt, timestamp
                Optional: velocity, accuracy, leo_sats, gnss_sats, etc.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connected or not self.client:
            logger.error("Not connected to Supabase")
            return False
        
        try:
            # Prepare data payload
            payload = {
                'session_id': self.session_id,
                'timestamp': nav_data.get('timestamp', datetime.utcnow().isoformat()),
                'latitude': nav_data['lat'],
                'longitude': nav_data['lon'],
                'altitude': nav_data.get('alt', 0.0),
                'velocity': nav_data.get('velocity', 0.0),
                'accuracy': nav_data.get('accuracy', None),
                'position_uncertainty': nav_data.get('position_uncertainty', None),
                'velocity_uncertainty': nav_data.get('velocity_uncertainty', None),
                
                # Satellite counts
                'leo_satellites_used': nav_data.get('leo_sats_used', 0),
                'leo_satellites_visible': nav_data.get('leo_sats_visible', 0),
                'gnss_satellites': nav_data.get('gnss_sats', 0),
                
                # Quality metrics
                'gnss_available': nav_data.get('gnss_available', False),
                'imu_available': nav_data.get('imu_available', False),
                'hdop': nav_data.get('hdop', None),
                
                # Cognitive weights
                'gnss_weight': nav_data.get('gnss_weight', None),
                'leo_weight': nav_data.get('leo_weight', None),
                
                # Clock estimates
                'clock_bias': nav_data.get('clock_bias', None),
                'clock_drift': nav_data.get('clock_drift', None),
                
                # Additional metadata
                'mode': nav_data.get('mode', 'hybrid'),  # gnss_only, leo_only, hybrid
                'environment': nav_data.get('environment', 'unknown'),  # open_sky, urban, indoor
            }
            
            # Insert into table
            response = self.client.table(self.table_name).insert(payload).execute()
            
            logger.debug(f"Inserted navigation data: {payload['latitude']:.6f}°, "
                        f"{payload['longitude']:.6f}°, {payload['altitude']:.1f}m")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert navigation data: {e}")
            return False
    
    def insert_batch(self, nav_data_list: List[Dict]) -> bool:
        """
        Insert multiple navigation records at once.
        
        Args:
            nav_data_list: List of navigation data dictionaries
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connected or not self.client:
            logger.error("Not connected to Supabase")
            return False
        
        try:
            payloads = []
            
            for nav_data in nav_data_list:
                payload = {
                    'session_id': self.session_id,
                    'timestamp': nav_data.get('timestamp', datetime.utcnow().isoformat()),
                    'latitude': nav_data['lat'],
                    'longitude': nav_data['lon'],
                    'altitude': nav_data.get('alt', 0.0),
                    'velocity': nav_data.get('velocity', 0.0),
                    'accuracy': nav_data.get('accuracy', None),
                    'position_uncertainty': nav_data.get('position_uncertainty', None),
                    'leo_satellites_used': nav_data.get('leo_sats_used', 0),
                    'gnss_satellites': nav_data.get('gnss_sats', 0),
                }
                payloads.append(payload)
            
            # Batch insert
            response = self.client.table(self.table_name).insert(payloads).execute()
            
            logger.info(f"Batch inserted {len(payloads)} navigation records")
            return True
            
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            return False
    
    def get_latest_position(self, session_id: Optional[str] = None) -> Optional[Dict]:
        """
        Retrieve the latest navigation position from database.
        
        Args:
            session_id: Session ID to query (default: current session)
        
        Returns:
            Navigation data dict or None if not found
        """
        if not self.connected or not self.client:
            logger.error("Not connected to Supabase")
            return None
        
        try:
            session = session_id or self.session_id
            
            response = (self.client.table(self.table_name)
                       .select("*")
                       .eq('session_id', session)
                       .order('timestamp', desc=True)
                       .limit(1)
                       .execute())
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve latest position: {e}")
            return None
    
    def get_session_history(self, session_id: Optional[str] = None, 
                           limit: int = 100) -> List[Dict]:
        """
        Retrieve navigation history for a session.
        
        Args:
            session_id: Session ID to query (default: current session)
            limit: Maximum number of records to return
        
        Returns:
            List of navigation data dicts
        """
        if not self.connected or not self.client:
            logger.error("Not connected to Supabase")
            return []
        
        try:
            session = session_id or self.session_id
            
            response = (self.client.table(self.table_name)
                       .select("*")
                       .eq('session_id', session)
                       .order('timestamp', desc=False)
                       .limit(limit)
                       .execute())
            
            return response.data if response.data else []
            
        except Exception as e:
            logger.error(f"Failed to retrieve session history: {e}")
            return []
    
    def insert_doppler_measurement(self, doppler_data: Dict) -> bool:
        """
        Insert raw Doppler measurement (for analysis/debugging).
        
        Args:
            doppler_data: Dict with satellite_name, doppler_hz, snr_db, etc.
        
        Returns:
            True if successful
        """
        if not self.connected or not self.client:
            return False
        
        try:
            payload = {
                'session_id': self.session_id,
                'timestamp': doppler_data.get('timestamp', datetime.utcnow().isoformat()),
                'satellite_name': doppler_data['satellite_name'],
                'doppler_hz': doppler_data['doppler_hz'],
                'doppler_predicted_hz': doppler_data.get('doppler_predicted_hz', None),
                'snr_db': doppler_data.get('snr_db', None),
                'power_db': doppler_data.get('power_db', None),
                'weight': doppler_data.get('weight', None),
                'is_anomaly': doppler_data.get('is_anomaly', False),
            }
            
            # Insert into doppler_measurements table (if exists)
            response = self.client.table('doppler_measurements').insert(payload).execute()
            
            return True
            
        except Exception as e:
            logger.debug(f"Doppler measurement insert skipped or failed: {e}")
            return False
    
    def create_tables_if_not_exist(self):
        """
        Create required database tables if they don't exist.
        Note: This requires service_role key with admin privileges.
        
        SQL to run in Supabase SQL Editor:
        
        CREATE TABLE IF NOT EXISTS navigation_data (
            id BIGSERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            latitude DOUBLE PRECISION NOT NULL,
            longitude DOUBLE PRECISION NOT NULL,
            altitude DOUBLE PRECISION,
            velocity DOUBLE PRECISION,
            accuracy DOUBLE PRECISION,
            position_uncertainty DOUBLE PRECISION,
            velocity_uncertainty DOUBLE PRECISION,
            leo_satellites_used INTEGER,
            leo_satellites_visible INTEGER,
            gnss_satellites INTEGER,
            gnss_available BOOLEAN,
            imu_available BOOLEAN,
            hdop DOUBLE PRECISION,
            gnss_weight DOUBLE PRECISION,
            leo_weight DOUBLE PRECISION,
            clock_bias DOUBLE PRECISION,
            clock_drift DOUBLE PRECISION,
            mode TEXT,
            environment TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE INDEX idx_session_timestamp ON navigation_data(session_id, timestamp DESC);
        CREATE INDEX idx_timestamp ON navigation_data(timestamp DESC);
        
        -- Enable Row Level Security (optional)
        ALTER TABLE navigation_data ENABLE ROW LEVEL SECURITY;
        
        -- Create policy to allow public read access (adjust as needed)
        CREATE POLICY "Enable read access for all users" 
        ON navigation_data FOR SELECT 
        USING (true);
        
        -- Create policy to allow insert for authenticated/anon users
        CREATE POLICY "Enable insert for all users" 
        ON navigation_data FOR INSERT 
        WITH CHECK (true);
        """
        logger.info("Please run the SQL schema in Supabase SQL Editor (see docstring)")


def test_connection():
    """Test Supabase connection."""
    try:
        client = SupabaseNavigationClient()
        client.connect()
        
        # Test insert
        test_data = {
            'lat': 12.9716,
            'lon': 77.5946,
            'alt': 889.4,
            'velocity': 1.5,
            'accuracy': 10.2,
            'leo_sats_used': 3,
            'gnss_sats': 8,
        }
        
        success = client.insert_navigation_data(test_data)
        
        if success:
            print("✓ Supabase connection test PASSED")
            print(f"  Session ID: {client.session_id}")
            
            # Retrieve latest
            latest = client.get_latest_position()
            if latest:
                print(f"  Latest position: {latest['latitude']:.6f}°, {latest['longitude']:.6f}°")
        else:
            print("✗ Supabase connection test FAILED")
        
        client.disconnect()
        
    except Exception as e:
        print(f"✗ Supabase test failed: {e}")
        print("  Make sure .env file is configured with SUPABASE_URL and SUPABASE_KEY")


if __name__ == '__main__':
    test_connection()
