sftp-fake:
	rsync -anv --exclude-from=".syncignore" $(PWD) lame23:/home/infres/dkiedanski
sftp:
	rsync -av --exclude-from=".syncignore" $(PWD) lame23:/home/infres/dkiedanski
