import asyncio
import aiohttp
from collections import defaultdict, deque
from functools import partial
import requests
import time
import json


import streamlit as st
st.set_page_config(page_title="stream", layout="wide")


WS_CONN = "ws://localhost:8000/airquality"
SW_CONN = "http://localhost:8000"

def prueba_envio_service_web():

    try:
        status = False
        data = '2'
        status = requests.post(SW_CONN + "/send-notification/"+data).status_code == 200
    except:
        return status
    return status



async def consumer_airquality( status):

    async with aiohttp.ClientSession(trust_env=True) as session:
        status.subheader(f"Connecting to {WS_CONN}")
        async with session.ws_connect(WS_CONN) as websocket:
            status.subheader(f"Connected to: {WS_CONN}")
            async for message in websocket:
                json_data = json.loads(message.data)
                print( type(message.type))
                if isinstance(json_data, list) and len(json_data) > 0:
                        # Access the first element of the JSON array
                        first_element = json_data[0]
                        my_bar.progress((int(first_element) + 1 )* 2, text=progress_text)
                        print(first_element)
                        st.write(first_element)
                #st.write(json_data)

    st.write("TERMINO")
                #   if msg.type == aiohttp.WSMsgType.TEXT:
                #     print(msg.data)
                #     if msg.data == 'close':
                #         await ws.close()

                # windows["raw"].append(data[3])
                # windows["graph"].append(data[4])
                # windows["map"].append({"lat": data[5], "lon": data[6]})

                # for column_name, graph in graphs.items():
                #     await asyncio.sleep(0.1)
                #     sensor_data = {column_name: windows[column_name]}
                #     if column_name == "raw":
                #         graph.write(data)
                #     if column_name == "graph":
                #         graph.line_chart(sensor_data)
                #     if column_name == "map":
                #         df = pd.DataFrame(
                #             [i for i in list(sensor_data["map"]) if i != 0]
                #         )
                #         graph.map(df, zoom=0)


status = st.empty()
connect = st.checkbox("Connect to WS Server")

progress_text = "Operation in progress. Please wait."

my_bar = st.progress(0, text=progress_text)

#for percent_complete in range(100):
    #time.sleep(0.1)
#    my_bar.progress(percent_complete + 1, text=progress_text)



if connect:
    prueba_envio_service_web()
    asyncio.run(
        consumer_airquality(
            status
        )
    )
else:
    status.subheader(f"Disconnected.")


