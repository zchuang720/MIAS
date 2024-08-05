# -*- coding: utf-8 -*-
import asyncio
import json
import os
import uuid
import websockets
from vosk import Model, KaldiRecognizer, SetLogLevel

SetLogLevel(0)

websocket_users = set()

vosk_model = Model("./model/asr/model/vosk-model-small-cn-0.22")


async def check_user_permit(websocket):
    print("new websocket_users:", websocket)
    websocket_users.add(websocket)
    print("websocket_users list:", websocket_users)
    while False:
        recv_str = await websocket.recv()
        cred_dict = recv_str.split(":")
        if cred_dict[0] == "admin" and cred_dict[1] == "123456":
            response_str = "Congratulation, you have connect with server..."
            await websocket.send(response_str)
            print("Password is ok...")
            return True
        else:
            response_str = "Sorry, please input the username or password..."
            print("Password is wrong...")
            await websocket.send(response_str)

# 接收客户端消息并处理，这里只是简单把客户端发来的返回回去
async def recv_user_msg(websocket):
    rec = KaldiRecognizer(vosk_model, 16000)
    rec.SetWords(False)
    uid = str(uuid.uuid4())
    suid = ''.join(uid.split('-'))

    while True:
        recv_text = await websocket.recv()
        if rec.AcceptWaveform(recv_text):
            t_text = rec.Result()
        else:
            t_text = rec.PartialResult()
        user_dic = json.loads(t_text)

        # if ("text" in user_dic):
        #     r_text = user_dic["text"] + ","
        # else:
        #     r_text = user_dic["partial"]
        user_dic['uid'] = suid  # 添加
        t = json.dumps(user_dic, ensure_ascii=False)
        print(t)
        await websocket.send(t)

# 服务器端主逻辑
async def run(websocket, path):
    while True:
        try:
            await check_user_permit(websocket)
            await recv_user_msg(websocket)
        except websockets.ConnectionClosed:
            print("ConnectionClosed...", path)  # 链接断开
            print("websocket_users old:", websocket_users)
            websocket_users.remove(websocket)
            print("websocket_users new:", websocket_users)
            break
        except websockets.InvalidState:
            print("InvalidState...")  # 无效状态
            break
        except Exception as e:
            print("Exception:", e)

def run_server():
    global websocket_users, vosk_model
    asyncio.get_event_loop().run_until_complete(websockets.serve(run, "0.0.0.0", 8081))
    asyncio.get_event_loop().run_forever()
    print("Real-Time ASR Model Started")

if __name__ == '__main__':
    run_server()
