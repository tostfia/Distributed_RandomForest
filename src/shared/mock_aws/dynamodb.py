from typing import Optional

class MockDynamoDB:
    def __init__(self):
        self._tables = {
            "ModelStatus": {},       
            "ServiceDiscovery": {}  
        }

    def put_item(self, table_name: str, key: str, value: dict):
        if table_name not in self._tables:
            self._tables[table_name] = {}
        self._tables[table_name][key] = value
        print(f"[Mock DynamoDB] Tabella '{table_name}' -> Scritto ID: {key[:8]} | Stato: {value.get('status')}")

    def get_item(self, table_name: str, key: str) -> Optional[dict]:
        if table_name in self._tables and key in self._tables[table_name]:
            return self._tables[table_name][key]
        return None

# Istanza globale
dynamo_db = MockDynamoDB()