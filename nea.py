__version__ = '0.6'


import sys
import string
import os
import cgi
import urllib
import socket
import re
import stat
import random
import time
import struct
import zipfile
import ezt
import MP3Info
import md5

try:
  import signal
  signalSupport = 'yes'
except ImportError:
  signalSupport = 'no'

try:
  import ogg.vorbis
  oggSupport = 'yes'
except ImportError:
  oggSupport = 'no'

try:
  import cStringIO
  StringIO = cStringIO
except ImportError:
  import StringIO

try:
  import sha
except ImportError:
  pass





class Server(mixin, BaseHTTPServer.HTTPServer):
  def __init__(self, fname):
    self.userLog = [ ] 
    self.userIPs = { } 

    config = self.config = ConfigParser.ConfigParser()

    config.add_section('server')
    config.add_section('sources')
    config.add_section('acl')
    config.add_section('extra')

    
