#!/bin/bash
rsync -av --exclude-from=.syncignore "$PWD" dkiedanski@lame23.enst.fr:/home/infres/dkiedanski/coop_extension_code
