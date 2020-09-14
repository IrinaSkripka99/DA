from __future__ import print_function, division, unicode_literals, absolute_import
from streamlit.compatibility import setup_2_3_shims

setup_2_3_shims(globals())

import streamlit as st
from streamlit import config

# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import SGD
# from keras.utils import np_utils
# from tensorflow import keras

import altair as alt
import pydeck as pdk
import time
from streamlit import config
import random
import plotly.graph_objs as go
from io import BytesIO
import requests
from random import random
import platform
import sys
from collections import namedtuple


import graphviz as graphviz
import pandas as pd
import os
import io
import math
import wave
import time
from scipy.io import wavfile
import numpy as np
import altair as alt
from datetime import datetime
import pydeck as pdk
import random
from datetime import date

from PIL import Image, ImageDraw

from streamlit.widgets import Widgets


DATE_TIME = "date/time"
DATA_URL = (
    "http://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz"
)

demo = st.sidebar.selectbox(
    "Choose demo", ["Altair", "Uber example", "Apocrypha", "Audio", "Checkboxes",
    'Empty charts','Graphiz','Images','Interactive widgets','Lists',"Animation", 
    "Caching", "Plotly example", "Reference","Run on save", "Syntax error", "Syntax hilite", "Video"
    ], 0
)

if demo == "Altair":
    df = pd.DataFrame(np.random.randn(200, 3), columns=["a", "b", "c"])
    c = (
        alt.Chart(df)
        .mark_circle()
        .encode(x="a", y="b", size="c", color="c")
        .interactive()
    )
    st.title("These two should look exactly the same")
    st.write("Altair chart using `st.altair_chart`:")
    st.altair_chart(c)
    st.write("And the same chart using `st.write`:")
    st.write(c)
if demo == "Uber example":
    """
    # Uber Pickups in New York City
    # """
    # """
    # """
    @st.cache(persist=True)
    def load_data(nrows):
        data = pd.read_csv(DATA_URL, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis="columns", inplace=True)
        data[DATE_TIME] = pd.to_datetime(data[DATE_TIME])
        return data

    data = load_data(100000)
    "data", data
    hour = st.sidebar.number_input("hour", 0, 23, 10)
    data = data[data[DATE_TIME].dt.hour == hour]

    "## Geo Data at %sh" % hour
    st.map(data)
    ""
    midpoint = (np.average(data["lat"]), np.average(data["lon"]))
    st.write(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state={
                "latitude": midpoint[0],
                "longitude": midpoint[1],
                "zoom": 11,
                "pitch": 50,
            },
            layers=[
                pdk.Layer(
                    "HexagonLayer",
                    data=data,
                    get_position=["lon", "lat"],
                    radius=100,
                    elevation_scale=4,
                    elevation_range=[0, 1000],
                    pickable=True,
                    extruded=True,
                ),
            ],
        )
    )
    if st.checkbox("Show Raw Data"):
        "## Raw Data at %sh" % hour, data
if demo == "Apocrypha":
    """The crypt of top secret undocumented Streamlit API calls."""

    setup_2_3_shims(globals())
    st.title("Apocrypha")
    st.write("The crypt of top secret _undocumented_ Streamlit API calls.")

    st.header("Tables")
    with st.echo():
        arrays = [
            np.array(["bar", "bar", "baz", "baz", "foo", None, "qux", "qux"]),
            np.array(["one", "two", "one", "two", "one", "two", "one", "two"]),
        ]

    df = pd.DataFrame(
        np.random.randn(8, 4),
        index=arrays,
        columns=[
            datetime(2012, 5, 1),
            datetime(2012, 5, 2),
            datetime(2012, 5, 3),
            datetime(2012, 5, 4),
        ],
    )

    st.subheader("A table")
    st.table(df)

    st.subheader("...and its transpose")
    st.table(df.T)
    st.header("Maps")
    st.warning("TODO: Need to document the st.map() API here.")
    st.balloons()
if demo == "Audio":
    st.title("Audio test")
    st.header("Local file")
    # These are the formats supported in Streamlit right now.
    AUDIO_EXTENSIONS = ["wav", "flac", "mp3", "aac", "ogg", "oga", "m4a", "opus", "wma"]

    def get_audio_files_in_dir(directory):
        out = []
        for item in os.listdir(directory):
            try:
                name, ext = item.split(".")
            except:
                continue
            if name and ext:
                if ext in AUDIO_EXTENSIONS:
                    out.append(item)
        return out

    avdir = os.path.expanduser("~")
    audiofiles = get_audio_files_in_dir(avdir)
    if len(audiofiles) == 0:
        st.write(
            "Put some audio files in your home directory (%s) to activate this player."
            % avdir
        )
    else:
        filename = st.selectbox(
            "Select an audio file from your home directory (%s) to play" % avdir,
            audiofiles,
            0,
        )
        audiopath = os.path.join(avdir, filename)
        st.audio(audiopath)

    st.header("Generated audio (440Hz sine wave)")

    def note(freq, length, amp, rate):
        t = np.linspace(0, length, length * rate)
        data = np.sin(2 * np.pi * freq * t) * amp
        return data.astype(np.int16)

    nchannels = 1
    frequency = 440
    hertznchannels = 1
    sampwidth = 2
    sampling_rate = 44100
    duration = 89  # Max size, given the bitrate and sample width
    comptype = "NONE"
    compname = "not compressed"
    amplitude = 10000
    nframes = duration * sampling_rate

    x = st.text("Making wave...")
    sine_wave = note(frequency, duration, amplitude, sampling_rate)

    fh = wave.open("sound.wav", "w")
    fh.setparams(
        (nchannels, sampwidth, int(sampling_rate), nframes, comptype, compname)
    )

    x.text("Converting wave...")
    fh.writeframes(sine_wave)

    fh.close()
    with io.open("sound.wav", "rb") as f:
        x.text("Sending wave...")
        x.audio(f)
        st.header("Audio from a Remote URL")

    def shorten_audio_option(opt):
        return opt.split("/")[-1]

    song = st.selectbox(
        "Pick an MP3 to play",
        (
            "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
            "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3",
            "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-8.mp3",
        ),
        0,
        shorten_audio_option,
    )

    st.audio(song)

    st.title("Streaming audio from a URL")

    st.write("[MP3: Mutiny Radio](http://nthmost.net:8000/mutiny-studio)")

    st.audio("http://nthmost.net:8000/mutiny-studio")

    st.write("[OGG: Radio Loki](http://nthmost.net:8000/loki.ogg)")

    st.audio("http://nthmost.net:8000/loki.ogg")
if demo == "Checkboxes":

    @st.cache
    def create_image(r=0, g=0, b=0, a=0):
        color = "rgb(%d%%, %d%%, %d%%)" % (int(r), int(g), int(b))
        
        size = 200
        step = 10
        half = size / 2

       # Create a new image
        image = Image.new("RGB", (size, size), color=color)
        d = ImageDraw.Draw(image)

       # Draw a red square
        d.rectangle(
            [(step, step), (half - step, half - step)], fill="red", outline=None, width=0
        )

       # Draw a green circle.  In PIL, green is 00800, lime is 00ff00
        d.ellipse(
            [(half + step, step), (size - step, half - step)],
            fill="lime",
            outline=None,
            width=0,
        )

    # Draw a blue triangle
        d.polygon(
           [(half / 2, half + step), (half - step, size - step), (step, size - step)],
           fill="blue",
           outline=None,
       )

    # Creating a pie slice shaped 'mask' ie an alpha channel.
        alpha = Image.new("L", image.size, 0xFF)
        d = ImageDraw.Draw(alpha)
        d.pieslice(
           [(step * 3, step * 3), (size - step, size - step)],
           0,
           90,
           fill=a,
           outline=None,
           width=0,
        )

        image.putalpha(alpha)

        return np.array(image).astype("float") / 255.0
    
    if True:
       st.title("Image, checkbox and slider test")

       st.write("Script ran at", datetime.now().isoformat())

       st.subheader("Background color")
       r_color = st.slider("Red amount", 0, 100)
       g_color = st.slider("Green amount", 0, 100)
       b_color = st.slider("Blue amount", 0, 100)
       alpha_pct = st.slider("Alpha amount", 0, 100, 50)

       image = create_image(r_color, g_color, b_color, alpha_pct)
       r = image[:, :, 0]
       g = image[:, :, 1]
       b = image[:, :, 2]
       alpha = image[:, :, 3]

       z = np.zeros(r.shape)
       mask = np.ones(r.shape)

       image = np.stack([r, g, b], 2)

       st.subheader("Channels to include in output")
       r_on = st.checkbox("Red", True)
       g_on = st.checkbox("Green", True)
       b_on = st.checkbox("Blue", True)
       alpha_on = st.checkbox("Alpha", True)
       image = np.stack(
           [
               r if r_on else z,
               g if g_on else z,
               b if b_on else z,
               alpha if alpha_on else mask,
           ],
           2,
       )

    st.image(image, format="png")
if demo == 'Empty charts':
    st.title("Empty charts")

    st.write(
        """
        This file tests what happens when you pass an empty dataframe or `None` into
        a chart.
        In some cases, we handle it nicely. In others, we show an error. The reason
        for the latter is because some chart types derive their configuration from
        the dataframe you pass in at the start. So when there's no dataframe we
        cannot detect that configuration.
    """
    )

    data = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 3, 2, 4]})

    spec = {
            "mark": "line",
            "encoding": {
            "x": {"field": "a", "type": "quantitative"},
            "y": {"field": "b", "type": "quantitative"},
        },
    }

    st.subheader("Here are 4 empty charts")
    st.vega_lite_chart(spec)
    st.line_chart()
    st.area_chart()
    st.bar_chart()

    st.write("Below is an empty pyplot chart (i.e. just a blank image)")
    st.pyplot()
    st.write("...and that was it.")

    st.subheader("Here are 5 filled charts")
    x = st.vega_lite_chart(spec)
    x.vega_lite_chart(data, spec)

    x = st.vega_lite_chart(spec)
    time.sleep(0.2)  # Sleep a little so the add_rows gets sent separately.
    x.add_rows(data)

    x = st.line_chart()
    x.add_rows(data)

    x = st.area_chart()
    x.add_rows(data)

    x = st.bar_chart()
    x.add_rows(data)

    st.subheader("Here is 1 empty map")
    st.deck_gl_chart()
if demo == 'Graphiz':
    """examples from https://graphviz.readthedocs.io/en/stable/examples.html"""

    # basic graph
    hello = graphviz.Digraph("Hello World")
    hello.edge("Hello", "World")

    # styled graph
    styled = graphviz.Graph("G", filename="g_c_n.gv")
    styled.attr(bgcolor="purple:pink", label="agraph", fontcolor="white")

    with styled.subgraph(name="cluster1") as c:
        c.attr(
            fillcolor="blue:cyan",
            label="acluster",
            fontcolor="white",
            style="filled",
            gradientangle="270",
        )
        c.attr(
            "node", shape="box", fillcolor="red:yellow", style="filled", gradientangle="90"
        )
        c.node("anode")

    # complex graph
    finite = graphviz.Digraph("finite_state_machine", filename="fsm.gv")
    finite.attr(rankdir="LR", size="8,5")

    finite.attr("node", shape="doublecircle")
    finite.node("LR_0")
    finite.node("LR_3")
    finite.node("LR_4")
    finite.node("LR_8")

    finite.attr("node", shape="circle")
    finite.edge("LR_0", "LR_2", label="SS(B)")
    finite.edge("LR_0", "LR_1", label="SS(S)")
    finite.edge("LR_1", "LR_3", label="S($end)")
    finite.edge("LR_2", "LR_6", label="SS(b)")
    finite.edge("LR_2", "LR_5", label="SS(a)")
    finite.edge("LR_2", "LR_4", label="S(A)")
    finite.edge("LR_5", "LR_7", label="S(b)")
    finite.edge("LR_5", "LR_5", label="S(a)")
    finite.edge("LR_6", "LR_6", label="S(b)")
    finite.edge("LR_6", "LR_5", label="S(a)")
    finite.edge("LR_7", "LR_8", label="S(b)")
    finite.edge("LR_7", "LR_5", label="S(a)")
    finite.edge("LR_8", "LR_6", label="S(b)")
    finite.edge("LR_8", "LR_5", label="S(a)")

    # draw graphs
    st.write("You should see a graph with two connected nodes.")
    st.graphviz_chart(hello)

    st.write("You should see a colorful node within a cluster within a graph.")
    st.graphviz_chart(styled)

    st.write("You should see a graph representing a finite state machine.")
    st.graphviz_chart(finite)
if demo == 'Images' :
    """St.image example."""
    class StreamlitImages(object):
        def __init__(self, size=200, step=10):
            self._size = size
            self._step = step
            self._half = self._size / 2
            self._data = {}

            self.create_image()
            self.generate_image_types()
            self.generate_image_channel_data()
            self.generate_bgra_image()
            self.generate_gif()
            self.generate_pseudorandom_image()
            self.generate_gray_image()
            self.save()

        def create_image(self):
            # Create a new image
            self._image = Image.new("RGB", (self._size, self._size))
            d = ImageDraw.Draw(self._image)

            # Draw a red square
            d.rectangle(
                [
                    (self._step, self._step),
                    (self._half - self._step, self._half - self._step),
                ],
                fill="red",
                outline=None,
                width=0,
            )

            # Draw a green circle.  In PIL, green is 00800, lime is 00ff00
            d.ellipse(
                [
                    (self._half + self._step, self._step),
                    (self._size - self._step, self._half - self._step),
                ],
                fill="lime",
                outline=None,
                width=0,
            )

            # Draw a blue triangle
            d.polygon(
                [
                    (self._half / 2, self._half + self._step),
                    (self._half - self._step, self._size - self._step),
                    (self._step, self._size - self._step),
                ],
                fill="blue",
                outline=None,
            )

            # Creating a pie slice shaped 'mask' ie an alpha channel.
            alpha = Image.new("L", self._image.size, "white")
            d = ImageDraw.Draw(alpha)
            d.pieslice(
                [
                    (self._step * 3, self._step * 3),
                    (self._size - self._step, self._size - self._step),
                ],
                0,
                90,
                fill="black",
                outline=None,
                width=0,
            )
            self._image.putalpha(alpha)

        def generate_image_types(self):
            for fmt in ("jpeg", "png"):
                i = self._image.copy()
                d = ImageDraw.Draw(i)
                d.text((self._step, self._step), fmt, fill=(0xFF, 0xFF, 0xFF, 0xFF))
                # jpegs dont have alpha channels.
                if fmt == "jpeg":
                    i = i.convert("RGB")
                data = io.BytesIO()
                i.save(data, format=fmt.upper())
                self._data["image.%s" % fmt] = data.getvalue()

        def generate_image_channel_data(self):
            # np.array(image) returns the following shape
            #   (width, height, channels)
            # and
            #   transpose((2, 0, 1)) is really
            #   transpose((channels, width, height))
            # So then we get channels, width, height which makes extracting
            # single channels easier.
            array = np.array(self._image).transpose((2, 0, 1))

            for idx, name in zip(range(0, 4), ["red", "green", "blue", "alpha"]):
                data = io.BytesIO()
                img = Image.fromarray(array[idx].astype(np.uint8))
                img.save(data, format="PNG")
                self._data["%s.png" % name] = data.getvalue()

        def generate_bgra_image(self):
            # Split Images and rearrange
            array = np.array(self._image).transpose((2, 0, 1))

            # Recombine image to BGRA
            bgra = (
                np.stack((array[2], array[1], array[0], array[3]))
                .astype(np.uint8)
                .transpose(1, 2, 0)
            )

            data = io.BytesIO()
            Image.fromarray(bgra).save(data, format="PNG")
            self._data["bgra.png"] = data.getvalue()

        def generate_gif(self):
            # Create grayscale image.
            im = Image.new("L", (self._size, self._size), "white")

            images = []

            # Make ten frames with the circle of a random size and location
            random.seed(0)
            for i in range(0, 10):
                frame = im.copy()
                draw = ImageDraw.Draw(frame)
                pos = (random.randrange(0, self._size), random.randrange(0, self._size))
                circle_size = random.randrange(10, self._size / 2)
                draw.ellipse([pos, tuple(p + circle_size for p in pos)], "black")
                images.append(frame.copy())

            # Save the frames as an animated GIF
            data = io.BytesIO()
            images[0].save(
                data,
                format="GIF",
                save_all=True,
                append_images=images[1:],
                duration=100,
                loop=0,
            )

            self._data["circle.gif"] = data.getvalue()

        def generate_pseudorandom_image(self):
            w, h = self._size, self._size
            r = np.array([255 * np.sin(x / w * 2 * np.pi) for x in range(0, w)])
            g = np.array([255 * np.cos(x / w * 2 * np.pi) for x in range(0, w)])
            b = np.array([255 * np.tan(x / w * 2 * np.pi) for x in range(0, w)])

            r = np.tile(r, h).reshape(w, h).astype("uint8")
            g = np.tile(g, h).reshape(w, h).astype("uint8")
            b = np.tile(b, h).reshape(w, h).astype("uint8")

            rgb = np.stack((r, g, b)).transpose(1, 2, 0)

            data = io.BytesIO()
            Image.fromarray(rgb).save(data, format="PNG")
            self._data["pseudorandom.png"] = data.getvalue()

        def generate_gray_image(self):
            gray = (
                np.tile(np.arange(self._size) / self._size * 255, self._size)
                .reshape(self._size, self._size)
                .astype("uint8")
            )

            data = io.BytesIO()
            Image.fromarray(gray).save(data, format="PNG")
            self._data["gray.png"] = data.getvalue()

        def save(self):
            for name, data in self._data.items():
                Image.open(io.BytesIO(data)).save("./tmp/%s" % name)

        def get_images(self):
            return self._data


    # Generate some images.
    si = StreamlitImages()

    # Get a single image of bytes and display
    st.header("individual image bytes")
    filename = "image.png"
    data = si.get_images().get(filename)
    st.image(data, caption=filename, format="png")

    # Display a list of images
    st.header("list images")
    images = []
    captions = []
    for filename, data in si.get_images().items():
        images.append(data)
        captions.append(filename)
    st.image(images, caption=captions, format="png")

    st.header("PIL Image")
    data = []

    # Get a single image to use for all the numpy stuff
    image = Image.open(io.BytesIO(si.get_images()["image.png"]))
    data.append((image, "PIL Image.open('image.png')"))
    image = Image.open(io.BytesIO(si.get_images()["image.jpeg"]))
    data.append((image, "PIL Image.open('image.jpeg')"))
    data.append(
        (Image.new("RGB", (200, 200), color="red"), "Image.new('RGB', color='red')")
    )

    images = []
    captions = []
    for i, c in data:
        images.append(i)
        captions.append(c)
    st.image(images, caption=captions, format="png")

    st.header("Bytes IO Image")
    image = io.BytesIO(si.get_images()["image.png"])
    st.image(image, caption=str(type(image)), format="png")

    st.header("From a file")
    st.image("./tmp/image.png", caption="./tmp/image.png", format="png")

    st.header("From open")
    st.image(open("./tmp/image.png", "rb").read(), caption="from read", format="png")

    st.header("Numpy arrays")
    image = Image.open(io.BytesIO(si.get_images()["image.png"]))
    rgba = np.array(image)

    data = []
    # Full RGBA image
    data.append((rgba, str(rgba.shape)))
    # Select second channel
    data.append((rgba[:, :, 1], str(rgba[:, :, 1].shape)))
    # Make it x, y, 1
    data.append(
        (np.expand_dims(rgba[:, :, 2], 2), str(np.expand_dims(rgba[:, :, 2], 2).shape))
    )
    # Drop alpha channel
    data.append((rgba[:, :, :3], str(rgba[:, :, :3].shape)))

    images = []
    captions = []
    for i, c in data:
        images.append(i)
        captions.append(c)
    st.image(images, caption=captions, format="png")

    try:
        st.header("opencv")
        import cv2

        image = np.fromstring(si.get_images()["image.png"], np.uint8)

        img = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        st.image(img, format="png")
    except Exception:
        pass

    st.header("url")
    url = "https://farm1.staticflickr.com/9/12668460_3e59ce4e61.jpg"
    st.image(url, caption=url, width=200)

    st.header("Clipping")
    data = []
    np.random.seed(0)
    g = np.zeros((200, 200))
    b = np.zeros((200, 200))

    a = (np.random.ranf(size=200)) * 255
    r = np.array(np.gradient(a)) / 255.0 + 0.5

    a = np.tile(r, 200).reshape((200, 200))
    a = a - 0.3
    rgb = np.stack((a, g, b)).transpose(1, 2, 0)
    data.append((rgb, "clamp: rgb image - 0.3"))

    a = np.tile(r, 200).reshape((200, 200))
    rgb = np.stack((a, g, b)).transpose(1, 2, 0)
    data.append((rgb, "rgb image"))

    a = np.tile(r, 200).reshape((200, 200))
    a = a + 0.5
    rgb = np.stack((a, g, b)).transpose(1, 2, 0)
    data.append((rgb, "clamp: rgb image + 0.5"))

    images = []
    captions = []
    for i, c in data:
        images.append(i)
        captions.append(c)
    st.image(images, caption=captions, clamp=True, format="png")
if demo == 'Interactive widgets':
    st.title("Interactive Widgets")

    st.subheader("Checkbox")
    w1 = st.checkbox("I am human", True)
    st.write(w1)

    if w1:
        st.write("Agreed")

    st.subheader("Slider")
    w2 = st.slider("Age", 0.0, 100.0, (32.5, 72.5), 0.5)
    st.write(w2)

    st.subheader("Textarea")
    w3 = st.text_area("Comments", "Streamlit is awesomeness!")
    st.write(w3)

    st.subheader("Button")
    w4 = st.button("Click me")
    st.write(w4)

    if w4:
        st.write("Hello, Interactive Streamlit!")

    st.subheader("Radio")
    options = ("female", "male")
    w5 = st.radio("Gender", options, 1)
    st.write(w5)

    st.subheader("Text input")
    w6 = st.text_input("Text input widget", "i iz input")
    st.write(w6)

    st.subheader("Selectbox")
    options = ("first", "second")
    w7 = st.selectbox("Options", options, 1)
    st.write(w7)

    st.subheader("Time Input")
    w8 = st.time_input("Set an alarm for", time(8, 45))
    st.write(w8)

    st.subheader("Date Input")
    w9 = st.date_input("A date to celebrate", date(2019, 7, 6))
    st.write(w9)

    st.subheader("File Uploader")
    w10 = st.file_uploader("Upload a CSV file", type="csv")
    if w10:
        import pandas as pd

        data = pd.read_csv(w10)
        st.write(data)
if demo =='Lists':
    """An example of monitoring a simple neural net as it trains."""
    # dynamically grow the memory used on the GPU
    # this option is fine on non gpus as well.
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = True

    # https://kobkrit.com/using-allow-growth-memory-option-in-tensorflow-and-keras-dc8c8081bc96
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf_config))


    class MyCallback(keras.callbacks.Callback):
        def __init__(self, x_test):
            self._x_test = x_test

        def on_train_begin(self, logs=None):
            st.header("Summary")
            self._summary_chart = st.area_chart()
            self._summary_stats = st.text("%8s :  0" % "epoch")
            st.header("Training Log")

        def on_epoch_begin(self, epoch, logs=None):
            self._ts = time.time()
            self._epoch = epoch
            st.subheader("Epoch %s" % epoch)
            self._epoch_chart = st.line_chart()
            self._epoch_progress = st.info("No stats yet.")
            self._epoch_summary = st.empty()

        def on_batch_end(self, batch, logs=None):
            if batch % 10 == 0:
                rows = {"loss": [logs["loss"]], "accuracy": [logs["accuracy"]]}
                self._epoch_chart.add_rows(rows)
            if batch % 100 == 99:
                rows = {"loss": [logs["loss"]], "accuracy": [logs["accuracy"]]}
                self._summary_chart.add_rows(rows)
            percent_complete = logs["batch"] * logs["size"] / self.params["samples"]
            self._epoch_progress.progress(math.ceil(percent_complete * 100))
            ts = time.time() - self._ts
            self._epoch_summary.text(
                "loss: %(loss)7.5f | accuracy: %(accuracy)7.5f | ts: %(ts)d"
                % {"loss": logs["loss"], "accuracy": logs["accuracy"], "ts": ts}
            )

        def on_epoch_end(self, epoch, logs=None):
            # st.write('**Summary**')
            indices = np.random.choice(len(self._x_test), 36)
            test_data = self._x_test[indices]
            prediction = np.argmax(self.model.predict(test_data), axis=1)
            st.image(1.0 - test_data, caption=prediction)
            summary = "\n".join(
                "%(k)8s : %(v)8.5f" % {"k": k, "v": v} for (k, v) in logs.items()
            )
            st.text(summary)
            self._summary_stats.text(
                "%(epoch)8s :  %(epoch)s\n%(summary)s"
                % {"epoch": epoch, "summary": summary}
            )


    st.title("MNIST CNN")

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    img_width = 28
    img_height = 28

    x_train = x_train.astype("float32")
    x_train /= 255.0
    x_test = x_test.astype("float32")
    x_test /= 255.0

    # reshape input data
    x_train = x_train.reshape(x_train.shape[0], img_width, img_height, 1)
    x_test = x_test.reshape(x_test.shape[0], img_width, img_height, 1)

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    # build model

    model = Sequential()
    layer_1_size = 10
    epochs = 3

    model.add(Conv2D(10, (5, 5), input_shape=(img_width, img_height, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(config.layer_2_size, (5, 5), input_shape=(img_width, img_height,1), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(8, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    show_terminal_output = not config.get_option("server.liveSave")
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        verbose=show_terminal_output,
        callbacks=[MyCallback(x_test)],
    )

    st.success("Finished training!")

    # model.save("convnet.h5")
if demo == "Caching":
    cache_was_hit = True


    @st.cache
    def check_if_cached():
        global cache_was_hit
        cache_was_hit = False


    @st.cache
    def my_func(arg1, arg2=None, *args, **kwargs):
        return random.randint(0, 2 ** 32)


    check_if_cached()

    if cache_was_hit:
        st.warning("You must clear your cache before you run this script!")
        st.write(
            """
            To clear the cache, press `C` then `Enter`. Then press `R` on this page
            to rerun.
        """
        )
    else:
        st.warning(
            """
            IMPORTANT: You should test rerunning this script (to get a failing
            test), then clearing the cache with the `C` shortcut and checking that
            the test passes again.
        """
        )

        st.subheader("Test that basic caching works")
        u = my_func(1, 2, dont_care=10)
        # u = my_func(1, 2, dont_care=11)
        v = my_func(1, 2, dont_care=10)
        if u == v:
            st.success("OK")
        else:
            st.error("Fail")

        st.subheader("Test that when you change arguments it's a cache miss")
        v = my_func(10, 2, dont_care=10)
        if u != v:
            st.success("OK")
        else:
            st.error("Fail")

        st.subheader("Test that when you change **kwargs it's a cache miss")
        v = my_func(10, 2, dont_care=100)
        if u != v:
            st.success("OK")
        else:
            st.error("Fail")

        st.subheader("Test that you can turn off caching")
        config.set_option("client.caching", False)
        v = my_func(1, 2, dont_care=10)
        if u != v:
            st.success("OK")
        else:
            st.error("Fail")

        st.subheader("Test that you can turn on caching")
        config.set_option("client.caching", True)


        # Redefine my_func because the st.cache-decorated function "remembers" the
        # config option from when it was declared.
        @st.cache
        def my_func(arg1, arg2=None, *args, **kwargs):
            return random.randint(0, 2 ** 32)


        u = my_func(1, 2, dont_care=10)
        v = my_func(1, 2, dont_care=10)
        if u == v:
            st.success("OK")
        else:
            st.error("Fail")
if demo == "Animation":
    st.empty()
    my_bar = st.progress(0)
    for i in range(100):
        my_bar.progress(i + 1)
        time.sleep(0.1)
    n_elts = int(time.time() * 10) % 5 + 3
    for i in range(n_elts):
        st.text("." * i)
    st.write(n_elts)
    for i in range(n_elts):
        st.text("." * i)
    st.success("done")
if demo == "Code":
    st.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={
            "latitude": midpoint[0],
            "longitude": midpoint[1],
            "zoom": 11,
            "pitch": 50,
        },
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=data,
                get_position=["lon", "lat"],
                radius=100,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
        ],
    ))
if demo == "Plotly example":
    st.title("Plotly examples")

    st.header("Chart with two lines")

    trace0 = go.Scatter(x=[1, 2, 3, 4], y=[10, 15, 13, 17])
    trace1 = go.Scatter(x=[1, 2, 3, 4], y=[16, 5, 11, 9])
    data = [trace0, trace1]
    st.write(data)

    ###

    st.header("Matplotlib chart in Plotly")

    import matplotlib.pyplot as plt

    f = plt.figure()
    arr = np.random.normal(1, 1, size=100)
    plt.hist(arr, bins=20)

    st.plotly_chart(f)

    ###

    st.header("3D plot")

    x, y, z = np.random.multivariate_normal(np.array([0, 0, 0]), np.eye(3), 400).transpose()

    trace1 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(
            size=12,
            color=z,  # set color to an array/list of desired values
            colorscale="Viridis",  # choose a colorscale
            opacity=0.8,
        ),
    )

    data = [trace1]
    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
    fig = go.Figure(data=data, layout=layout)

    st.write(fig)

    ###

    st.header("Fancy density plot")

    import plotly.figure_factory as ff

    import numpy as np

    # Add histogram data
    x1 = np.random.randn(200) - 2
    x2 = np.random.randn(200)
    x3 = np.random.randn(200) + 2

    # Group data together
    hist_data = [x1, x2, x3]

    group_labels = ["Group 1", "Group 2", "Group 3"]

    # Create distplot with custom bin_size
    fig = ff.create_distplot(hist_data, group_labels, bin_size=[0.1, 0.25, 0.5])

    # Plot!
    st.plotly_chart(fig)
if demo == "Reference":
    st.title("Streamlit Quick Reference")

    st.header("The Basics")

    st.write("Import streamlit with `import streamlit as st`.")

    with st.echo():
        st.write(
            """
            The `write` function is Streamlit\'s bread and butter. You can use
            it to write _markdown-formatted_ text in your Streamlit app.
        """
        )

    with st.echo():
        the_meaning_of_life = 40 + 2

        st.write(
            "You can also pass in comma-separated values into `write` just like "
            "with Python's `print`. So you can easily interpolate the values of "
            "variables like this: ",
            the_meaning_of_life,
        )

    st.header("Visualizing data as tables")

    st.write(
        "The `write` function also knows what to do when you pass a NumPy "
        "array or Pandas dataframe."
    )

    with st.echo():
        import numpy as np

        a_random_array = np.random.randn(200, 200)

        st.write("Here's a NumPy example:", a_random_array)

    st.write("And here is a dataframe example:")

    with st.echo():
        import pandas as pd
        from datetime import datetime

        arrays = [
            np.array(["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"]),
            np.array(["one", "two", "one", "two", "one", "two", "one", None]),
        ]

        df = pd.DataFrame(
            np.random.randn(8, 4),
            index=arrays,
            columns=[
                datetime(2012, 5, 1),
                datetime(2012, 5, 2),
                datetime(2012, 5, 3),
                datetime(2012, 5, 4),
            ],
        )

        st.write(df, "...and its transpose:", df.T)

    st.header("Visualizing data as charts")

    st.write(
        "Charts are just as simple, but they require us to introduce some "
        "special functions first."
    )

    st.write("So assuming `data_frame` has been defined as...")

    with st.echo():
        chart_data = pd.DataFrame(
            np.random.randn(20, 5), columns=["pv", "uv", "a", "b", "c"]
        )

    st.write("...you can easily draw the charts below:")

    st.subheader("Example of line chart")

    with st.echo():
        st.line_chart(chart_data)

    st.write(
        "As you can see, each column in the dataframe becomes a different "
        "line. Also, values on the _x_ axis are the dataframe's indices. "
        "Which means we can customize them this way:"
    )

    with st.echo():
        chart_data2 = pd.DataFrame(
            np.random.randn(20, 2),
            columns=["stock 1", "stock 2"],
            index=pd.date_range("1/2/2011", periods=20, freq="M"),
        )

        st.line_chart(chart_data2)

    st.subheader("Example of area chart")

    with st.echo():
        st.area_chart(chart_data)

    st.subheader("Example of bar chart")

    with st.echo():
        trimmed_data = chart_data[["pv", "uv"]].iloc[:10]
        st.bar_chart(trimmed_data)

    st.subheader("Matplotlib")

    st.write(
        "You can use Matplotlib in Streamlit. "
        "Just use `st.pyplot()` instead of `plt.show()`."
    )
    try:
        # noqa: F401
        with st.echo():
            from matplotlib import cm, pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            # Create some data
            X, Y = np.meshgrid(np.arange(-5, 5, 0.25), np.arange(-5, 5, 0.25))
            Z = np.sin(np.sqrt(X ** 2 + Y ** 2))

            # Plot the surface.
            fig = plt.figure()
            ax = fig.gca(projection="3d")
            ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)

            st.pyplot()
    except Exception as e:
        err_str = str(e)
        if err_str.startswith("Python is not installed as a framework."):
            err_str = (
                "Matplotlib backend is not compatible with your Python "
                'installation. Please consider adding "backend: TkAgg" to your '
                " ~/.matplitlib/matplotlibrc. For more information, please see "
                '"Working with Matplotlib on OSX" in the Matplotlib FAQ.'
            )
        st.warning("Error running matplotlib: " + err_str)

    st.subheader("Vega-Lite")

    st.write(
        "For complex interactive charts, you can use "
        "[Vega-Lite](https://vega.github.io/vega-lite/):"
    )

    with st.echo():
        df = pd.DataFrame(np.random.randn(200, 3), columns=["a", "b", "c"])

        st.vega_lite_chart(
            df,
            {
                "mark": "circle",
                "encoding": {
                    "x": {"field": "a", "type": "quantitative"},
                    "y": {"field": "b", "type": "quantitative"},
                    "size": {"field": "c", "type": "quantitative"},
                    "color": {"field": "c", "type": "quantitative"},
                },
                # Add zooming/panning:
                "selection": {"grid": {"type": "interval", "bind": "scales"}},
            },
        )

    st.header("Visualizing data as images via Pillow.")


    @st.cache(persist=True)
    def read_file_from_url(url):
        try:
            return requests.get(url).content
        except requests.exceptions.RequestException:
            st.error("Unable to load file from %s. " "Is the internet connected?" % url)
        except Exception as e:
            st.exception(e)
        return None


    image_url = (
        "https://images.fineartamerica.com/images/artworkimages/"
        "mediumlarge/1/serene-sunset-robert-bynum.jpg"
    )
    image_bytes = read_file_from_url(image_url)

    if image_bytes is not None:
        with st.echo():
            # We can pass URLs to st.image:
            st.image(image_url, caption="Sunset", use_column_width=True)

            # For some reason, `PIL` requires you to import `Image` this way.
            from PIL import Image

            image = Image.open(BytesIO(image_bytes))

            array = np.array(image).transpose((2, 0, 1))
            channels = array.reshape(array.shape + (1,))

            # st.image also accepts byte arrays:
            st.image(channels, caption=["Red", "Green", "Blue"], width=200)

    st.header("Visualizing data as images via OpenCV")

    st.write("Streamlit also supports OpenCV!")
    try:
        import cv2

        if image_bytes is not None:
            with st.echo():
                image = cv2.cvtColor(
                    cv2.imdecode(np.fromstring(image_bytes, dtype="uint8"), 1),
                    cv2.COLOR_BGR2RGB,
                )

                st.image(image, caption="Sunset", use_column_width=True)
                st.image(cv2.split(image), caption=["Red", "Green", "Blue"], width=200)
    except ImportError as e:
        st.write(
            "If you install opencv with the command `pip install opencv-python-headless` "
            "this section will tell you how to use it."
        )

        st.warning("Error running opencv: " + str(e))

    st.header("Inserting headers")

    st.write(
        "To insert titles and headers like the ones on this page, use the `title`, "
        "`header`, and `subheader` functions."
    )

    st.header("Preformatted text")

    with st.echo():
        st.text(
            "Here's preformatted text instead of _Markdown_!\n"
            "       ^^^^^^^^^^^^\n"
            "Rock on! \m/(^_^)\m/ "
        )

    st.header("JSON")

    with st.echo():
        st.json({"hello": "world"})

    with st.echo():
        st.json('{"object":{"array":[1,true,"3"]}}')

    st.header("Inline Code Blocks")

    with st.echo():
        with st.echo():
            st.write("Use `st.echo()` to display inline code blocks.")

    st.header("Alert boxes")

    with st.echo():
        st.error("This is an error message")
        st.warning("This is a warning message")
        st.info("This is an info message")
        st.success("This is a success message")

    st.header("Progress Bars")

    with st.echo():
        for percent in [0, 25, 50, 75, 100]:
            st.write("%s%% progress:" % percent)
            st.progress(percent)

    st.header("Help")

    with st.echo():
        st.help(dir)

    st.header("Out-of-Order Writing")

    st.write("Placeholders allow you to draw items out-of-order. For example:")

    with st.echo():
        st.text("A")
        placeholder = st.empty()
        st.text("C")
        placeholder.text("B")

    st.header("Exceptions")
    st.write("You can print out exceptions using `st.exception()`:")

    with st.echo():
        try:
            raise RuntimeError("An exception")
        except Exception as e:
            st.exception(e)

    st.header("Playing audio")

    audio_url = (
        "https://upload.wikimedia.org/wikipedia/commons/c/c4/"
        "Muriel-Nguyen-Xuan-Chopin-valse-opus64-1.ogg"
    )
    audio_bytes = read_file_from_url(audio_url)

    st.write(
        """
        Streamlit can play audio in all formats supported by modern
        browsers. Below is an example of an _ogg_-formatted file:
        """
    )

    if audio_bytes is not None:
        with st.echo():
            st.audio(audio_bytes, format="audio/ogg")

    st.header("Playing video")

    st.write(
        """
        Streamlit can play video in all formats supported by modern
        browsers. Below is an example of an _mp4_-formatted file:
        """
    )

    video_url = "https://archive.org/download/WildlifeSampleVideo/" "Wildlife.mp4"
    video_bytes = read_file_from_url(video_url)

    if video_bytes is not None:
        with st.echo():
            st.video(video_bytes, format="video/mp4")

    st.header("Lengthy Computations")
    st.write(
        """
        If you're repeatedly running length computations, try caching the
        solution.
        ```python
        @streamlit.cache
        def lengthy_computation(...):
            ...
        # This runs quickly.
        answer = lengthy_computation(...)
        ```
        **Note**: `@streamlit.cache` requires that the function output
        depends *only* on its input arguments. For example, you can cache
        calls to API endpoints, but only do so if the data you get won't change.
    """
    )
    st.subheader("Spinners")
    st.write("A visual way of showing long computation is with a spinner:")


    def lengthy_computation():
        pass  # noop for demsontration purposes.


    with st.echo():
        with st.spinner("Computing something time consuming..."):
            lengthy_computation()

    st.header("Animation")
    st.write(
        """
        Every Streamlit method (except `st.write`) returns a handle
        which can be used for animation. Just call your favorite
        Streamlit function (e.g. `st.xyz()`) on the handle (e.g. `handle.xyz()`)
        and it will update that point in the app.
        Additionally, you can use `add_rows()` to append numpy arrays or
        DataFrames to existing elements.
    """
    )

    with st.echo():
        import numpy as np
        import time

        bar = st.progress(0)
        complete = st.text("0% complete")
        graph = st.line_chart()

        for i in range(100):
            bar.progress(i + 1)
            complete.text("%i%% complete" % (i + 1))
            graph.add_rows(np.random.randn(1, 2))

            time.sleep(0.1)

    st.title("Test of run-on-save")
    secs_to_wait = 5

    """
    How to test this:
    """

    st.info(
        """
        **First of all, make sure you're running the dev version of Streamlit** or
        that this file lives outside the Streamlit distribution. Otherwise, changes
        to this file may be ignored!
    """
    )

    """
    1. If run-on-save is on, make sure the page changes every few seconds. Then
       turn run-on-save off in the settigns menu and check (2).
    2. If run-on-save is off, make sure "Rerun"/"Always rerun" buttons appear in
       the status area. Click "Always rerun" and check (1).
    """

    st.write("This should change every ", secs_to_wait, " seconds: ", random())

    # Sleep for 5s (rather than, say, 1s) because on the first run we need to make
    # sure Streamlit is fully initialized before the timer below expires. This can
    # take several seconds.
    status = st.empty()
    for i in range(secs_to_wait, 0, -1):
        time.sleep(1)
        status.text("Sleeping %ss..." % i)

    status.text("Touching %s" % __file__)

    platform_system = platform.system()

    if platform_system == "Linux":
        cmd = (
                "sed -i "
                "'s/^# MODIFIED AT:.*/# MODIFIED AT: %(time)s/' %(file)s"
                " && touch %(file)s"
                % {  # sed on Linux modifies a different file.
                    "time": time.time(),
                    "file": __file__,
                }
        )

    elif platform_system == "Darwin":
        cmd = "sed -i bak " "'s/^# MODIFIED AT:.*/# MODIFIED AT: %s/' %s" % (
            time.time(),
            __file__,
        )

    # elif platform_system == "Windows":
    #     raise NotImplementedError("Windows not supported")
    #
    # else:
    #     raise Exception("Unknown platform")
    #
    # os.system(cmd)
    #
    # status.text("Touched %s" % __file__)

    # MODIFIED AT: 1580332945.720056
if demo == "Syntax error":
    st.title("Syntax error test")

    st.info("Uncomment the comment blocks in the source code one at a time.")

    st.write(
        """
        Here's the source file for you to edit:
        ```
        examples/syntax_error.py
        ```
        """
    )

    st.write("(Some top text)")

    # # Uncomment this as a block.
    # a = not_a_real_variable  # EXPECTED: inline exception.

    # # Uncomment this as a block.
    # if True  # EXPECTED: modal dialog

    # # Uncomment this as a block.
    # sys.stderr.write('Hello!\n')  # You should not see this.
    # # The line below is a compile-time error. Bad indentation.
    #        this_indentation_is_wrong = True  # EXPECTED: modal dialog

    # # Uncomment this as a block.
    # # This tests that errors after the first st call get caught.
    # a = not_a_real_variable  # EXPECTED: inline exception.

    st.write("(Some bottom text)")
if demo == "Syntax hilite":
    Language = namedtuple("Language", ["name", "example"])

    languages = [
        Language(
            name="Python",
            example="""
    # Python
    def say_hello():
        name = 'Streamlit'
        print('Hello, %s!' % name)""",
        ),
        Language(
            name="C",
            example="""
    /* C */
    int main(void) {
        const char *name = "Streamlit";
        printf(\"Hello, %s!\", name);
        return 0;
    }""",
        ),
        Language(
            name="JavaScript",
            example="""
    /* JavaScript */
    function sayHello() {
        const name = 'Streamlit';
        console.log(`Hello, ${name}!`);
    }""",
        ),
        Language(
            name="Shell",
            example="""
    # Bash/Shell
    NAME="Streamlit"
    echo "Hello, ${NAME}!"
    """,
        ),
        Language(
            name="SQL",
            example="""
    /* SQL */
    SELECT * FROM software WHERE name = 'Streamlit';
    """,
        ),
        Language(
            name="JSON",
            example="""
    {
        "_comment": "This is a JSON file!",
        name: "Streamlit",
        version: 0.27
    }""",
        ),
        Language(
            name="YAML",
            example="""
    # YAML
    software:
        name: Streamlit
        version: 0.27
    """,
        ),
        Language(
            name="HTML",
            example="""
    <!-- HTML -->
    <head>
      <title>Hello, Streamlit!</title>
    </head>
    """,
        ),
        Language(
            name="CSS",
            example="""
    /* CSS */
    .style .token.string {
        color: #9a6e3a;
        background: hsla(0, 0%, 100%, .5);
    }
    """,
        ),
        Language(
            name="JavaScript",
            example="""
    console.log('This is an extremely looooooooooooooooooooooooooooooooooooooooooooooooooooong string.')
        """,
        ),
    ]

    st.header("Syntax hiliting")

    st.subheader("Languages")
    for lang in languages:
        st.code(lang.example, lang.name)

    st.subheader("Other stuff")
    with st.echo():
        print("I'm inside an st.echo() block!")

    st.markdown(
        """
    This is a _markdown_ block...
    ```python
    print('...and syntax hiliting works here, too')
    ```
    """
    )
if demo == "Video":
    VIDEO_EXTENSIONS = ["mp4", "ogv", "m4v", "webm"]

    # For sample video files, try the Internet Archive, or download a few samples here:
    # http://techslides.com/sample-webm-ogg-and-mp4-video-files-for-html5

    st.title("Video Widget Examples")

    st.header("Local video files")
    st.write(
        "You can use st.video to play a locally-stored video by supplying it with a valid filesystem path."
    )


    def get_video_files_in_dir(directory):
        out = []
        for item in os.listdir(directory):
            try:
                name, ext = item.split(".")
            except:
                continue
            if name and ext:
                if ext in VIDEO_EXTENSIONS:
                    out.append(item)
        return out


    avdir = os.path.expanduser("~")
    files = get_video_files_in_dir(avdir)

    if len(files) == 0:
        st.write(
            "Put some video files in your home directory (%s) to activate this player."
            % avdir
        )

    else:
        filename = st.selectbox(
            "Select a video file from your home directory (%s) to play" % avdir, files, 0,
        )

        st.video(os.path.join(avdir, filename))
    st.header("Remote video playback")
    st.write("st.video allows a variety of HTML5 supported video links, including YouTube.")


    def shorten_vid_option(opt):
        return opt.split("/")[-1]


    # A random sampling of videos found around the web.  We should replace
    # these with those sourced from the streamlit community if possible!
    vidurl = st.selectbox(
        "Pick a video to play",
        (
            "https://youtu.be/_T8LGqJtuGc",
            "https://www.youtube.com/watch?v=kmfC-i9WgH0",
            "https://www.youtube.com/embed/sSn4e1lLVpA",
            "http://www.rochikahn.com/video/videos/zapatillas.mp4",
            "http://www.marmosetcare.com/video/in-the-wild/intro.webm",
            "https://www.orthopedicone.com/u/home-vid-4.mp4",
        ),
        0,
        shorten_vid_option,
    )
