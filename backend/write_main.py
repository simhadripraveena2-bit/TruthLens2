with open('main.py', 'w', encoding='utf-8') as f:
    f.write(open('main.py').read().replace(
        '@app.get(\"/annotate/next\")',
        '''@app.get(\"/health\")
async def health():
    import requests as r
    try:
        res = r.get(\"http://localhost:11434/api/tags\", timeout=5)
        models = [m[\"name\"] for m in res.json().get(\"models\", [])]
        return {\"status\": \"running\", \"ollama\": \"connected\", \"models\": models, \"gemma_available\": \"gemma3:4b\" in models}
    except Exception as e:
        return {\"status\": \"running\", \"ollama\": str(e)}

@app.get(\"/test\")
async def test():
    from analyzer import analyze_text
    try:
        result = analyze_text(\"Einstein was born in Germany. He invented the television.\", fast_mode=True)
        return {\"status\": \"success\", \"result\": result}
    except Exception as e:
        return {\"status\": \"error\", \"error\": str(e)}

@app.get(\"/annotate/next\")'''
    ))
print(\"Done!\")
