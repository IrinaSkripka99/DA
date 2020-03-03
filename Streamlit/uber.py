from __future__ import print_function, division, unicode_literals, absolute_import
from streamlit.compatibility import setup_2_3_shims

setup_2_3_shims(globals())

import streamlit as st
from streamlit import config

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from keras.utils import np_utils
from tensorflow import keras

import tensorflow as tf
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
from datetime import time
from datetime import date

from PIL import Image, ImageDraw

from streamlit.widgets import Widgets


DATE_TIME = "date/time"
DATA_URL = (
    "http://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz"
)

demo = st.sidebar.selectbox(
    "Choose demo", ["Altair", "Uber example", "Apocrypha", "Audio", "Checkboxes",
    'Empty charts','Graphiz','Images','Interactive widgets','Lists','Mnist cnn'
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
    st.title("Lists!")

    lists = [
        [],
        [10, 20, 30],
        [[10, 20, 30], [1, 2, 3]],
        [[10, 20, 30], [1]],
        [[10, "hi", 30], [1]],
        [[{"foo": "bar"}, "hi", 30], [1]],
        [[{"foo": "bar"}, "hi", 30], [1, [100, 200, 300, 400]]],
    ]


    for i, l in enumerate(lists):
        st.header("List %d" % i)

        st.write("With st.write")
        st.write(l)

        st.write("With st.json")
        st.json(l)

        st.write("With st.dataframe")
        st.dataframe(l)
if demo =='Mnist cnn':
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

