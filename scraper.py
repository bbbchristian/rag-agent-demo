"""
Web Scraper - Vehicle Parameter Data Collector
Usage: python scraper.py
"""
import sqlite3, os

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "vehicle_params.db")

def get_data():
    return [
        {"brand":"BMW","model":"i4","carline":"G26","system":"BCM","param_name":"DRL_Enable","param_group":"Lighting","data_type":"BOOLEAN","unit":"","min_val":0,"max_val":1,"default":"1","description":"Daytime running light enable"},
        {"brand":"BMW","model":"i4","carline":"G26","system":"BCM","param_name":"Headlamp_Leveling","param_group":"Lighting","data_type":"UINT8","unit":"level","min_val":0,"max_val":5,"default":"0","description":"Headlight level adjustment"},
        {"brand":"BMW","model":"i4","carline":"G26","system":"GW","param_name":"CAN_BaudRate","param_group":"Network","data_type":"UINT16","unit":"kbps","min_val":125,"max_val":1000,"default":"500","description":"CAN bus baud rate"},
        {"brand":"BMW","model":"i4","carline":"G26","system":"EPS","param_name":"SteeringTorque_Offset","param_group":"Chassis","data_type":"FLOAT","unit":"Nm","min_val":-5.0,"max_val":5.0,"default":"0.0","description":"Steering torque compensation"},
        {"brand":"Tesla","model":"Model 3","carline":"Highland","system":"BCM","param_name":"AmbientLight_Color","param_group":"Lighting","data_type":"UINT8","unit":"index","min_val":0,"max_val":15,"default":"3","description":"Ambient light color index"},
        {"brand":"Tesla","model":"Model 3","carline":"Highland","system":"IPU","param_name":"RegenBrake_Strength","param_group":"Chassis","data_type":"UINT8","unit":"%","min_val":0,"max_val":100,"default":"50","description":"Regenerative braking strength"},
        {"brand":"Tesla","model":"Model 3","carline":"Highland","system":"HU","param_name":"Volume_MaxLimit","param_group":"Infotainment","data_type":"UINT8","unit":"%","min_val":0,"max_val":100,"default":"80","description":"Max volume limit"},
        {"brand":"Tesla","model":"Model 3","carline":"Highland","system":"AC","param_name":"TargetTemp_Eco","param_group":"HVAC","data_type":"UINT8","unit":"degC","min_val":16,"max_val":30,"default":"22","description":"Eco mode target temperature"},
        {"brand":"VW","model":"ID.4","carline":"E371","system":"BCM","param_name":"FollowMeHome_Delay","param_group":"Lighting","data_type":"UINT16","unit":"s","min_val":0,"max_val":120,"default":"30","description":"Follow-me-home delay time"},
        {"brand":"VW","model":"ID.4","carline":"E371","system":"ESP","param_name":"ABS_Threshold","param_group":"Chassis","data_type":"FLOAT","unit":"","min_val":0.1,"max_val":1.0,"default":"0.8","description":"ABS trigger threshold"},
        {"brand":"VW","model":"ID.4","carline":"E371","system":"GW","param_name":"Gateway_SleepTimer","param_group":"Network","data_type":"UINT32","unit":"s","min_val":0,"max_val":3600,"default":"30","description":"Gateway sleep wait time"},
        {"brand":"VW","model":"ID.4","carline":"E371","system":"HU","param_name":"Display_Brightness","param_group":"Infotainment","data_type":"UINT8","unit":"%","min_val":0,"max_val":100,"default":"80","description":"Display brightness"},
        {"brand":"NIO","model":"ET5","carline":"NT2.0","system":"IPU","param_name":"TractionControl_Level","param_group":"Chassis","data_type":"UINT8","unit":"level","min_val":0,"max_val":3,"default":"1","description":"Traction control level"},
        {"brand":"NIO","model":"ET5","carline":"NT2.0","system":"GW","param_name":"Ethernet_WakeUp","param_group":"Network","data_type":"BOOLEAN","unit":"","min_val":0,"max_val":1,"default":"1","description":"Ethernet wake-up enable"},
        {"brand":"NIO","model":"ET5","carline":"NT2.0","system":"HU","param_name":"EQ_Preset","param_group":"Infotainment","data_type":"UINT8","unit":"","min_val":0,"max_val":6,"default":"1","description":"Equalizer preset mode"},
        {"brand":"BYD","model":"Seal","carline":"E321","system":"BCM","param_name":"DRL_Enable","param_group":"Lighting","data_type":"BOOLEAN","unit":"","min_val":0,"max_val":1,"default":"1","description":"DRL enable"},
        {"brand":"BYD","model":"Seal","carline":"E321","system":"AC","param_name":"AC_Compressor_MinOnTime","param_group":"HVAC","data_type":"UINT16","unit":"s","min_val":10,"max_val":300,"default":"60","description":"AC compressor min runtime"},
        {"brand":"BYD","model":"Seal","carline":"E321","system":"HU","param_name":"BT_DeviceLimit","param_group":"Infotainment","data_type":"UINT8","unit":"","min_val":1,"max_val":10,"default":"5","description":"Bluetooth device limit"},
        {"brand":"Mercedes","model":"EQS","carline":"V297","system":"EPS","param_name":"SteeringTorque_Offset","param_group":"Chassis","data_type":"FLOAT","unit":"Nm","min_val":-5.0,"max_val":5.0,"default":"0.0","description":"Steering torque offset"},
        {"brand":"Mercedes","model":"EQS","carline":"V297","system":"HU","param_name":"Language_Setting","param_group":"Infotainment","data_type":"UINT8","unit":"","min_val":0,"max_val":10,"default":"0","description":"Language setting (0=Chinese 1=English)"},
    ]

def init_db(records=None):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS vehicle_parameters")
    c.execute("""CREATE TABLE vehicle_parameters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        brand TEXT, model TEXT, carline TEXT, system TEXT,
        param_name TEXT, param_group TEXT, data_type TEXT, unit TEXT,
        min_val REAL, max_val REAL, default_val TEXT, description TEXT
    )""")
    if records is None:
        records = get_data()
    for r in records:
        c.execute("INSERT INTO vehicle_parameters (brand,model,carline,system,param_name,param_group,data_type,unit,min_val,max_val,default_val,description) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                  (r["brand"],r["model"],r["carline"],r["system"],r["param_name"],r["param_group"],r["data_type"],r["unit"],r["min_val"],r["max_val"],r["default"],r["description"]))
    conn.commit()
    cnt = c.execute("SELECT COUNT(*) FROM vehicle_parameters").fetchone()[0]
    conn.close()
    return cnt

if __name__ == "__main__":
    print("Initializing database...")
    cnt = init_db()
    print(f"Done! {cnt} records in database.")
