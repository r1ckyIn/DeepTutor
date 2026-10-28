"""OpenAI-style POST /v1/audio/speech on top of edge-tts, for DeepTutor's TTS.

DeepTutor's "OpenAI Compatible" voice adapter sends model, input, voice,
response_format and speed. The voice is an Edge voice name such as
zh-CN-XiaoxiaoNeural; model is ignored, and Edge returns mp3 only.
"""

from aiohttp import web
import edge_tts


async def speech(request: web.Request) -> web.Response:
    body = await request.json()
    if body.get("response_format", "mp3") != "mp3":
        raise web.HTTPBadRequest(text="only response_format=mp3 is supported")
    # OpenAI's speed is a multiplier; Edge wants a signed percentage.
    rate = f"{round((float(body.get('speed') or 1) - 1) * 100):+d}%"
    tts = edge_tts.Communicate(body["input"], body["voice"], rate=rate)
    audio = b"".join([c["data"] async for c in tts.stream() if c["type"] == "audio"])
    return web.Response(body=audio, content_type="audio/mpeg")


app = web.Application()
app.router.add_post("/v1/audio/speech", speech)
web.run_app(app, port=8000)
