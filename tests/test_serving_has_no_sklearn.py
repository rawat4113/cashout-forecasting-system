"""The real-time serving path must run with numpy + pandas only (no scikit-learn / joblib)."""
import subprocess
import sys
import textwrap


def test_realtime_path_without_sklearn(tmp_path):
    code = textwrap.dedent(f"""
        import sys
        sys.modules['sklearn'] = None; sys.modules['joblib'] = None     # any import of them now raises
        import os; os.environ['AUDIT_LOG_PATH'] = r'{tmp_path}/a.jsonl'
        from fastapi.testclient import TestClient
        import api
        c = TestClient(api.app)
        r = c.post('/api/v1/live/forecast', json={{'date': '2025-06-02', 'kind': 'OFFICIAL'}})
        assert r.status_code == 200, r.text
        assert c.get('/api/v1/audit/verify').json()['ok']
        assert c.get('/api/v1/forecast?date=2025-06-24&top_k=3').status_code == 503   # batch path degrades cleanly
        print('OK')
    """)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0 and "OK" in res.stdout, res.stderr[-800:]
