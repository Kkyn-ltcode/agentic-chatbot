Viewed download_datasets.sh:1-93

If you want to run PIDSMaker natively on a Linux VM (like via MobaXterm) without Docker, it is completely doable! It just means you have to manually install PostgreSQL and the Python dependencies yourself.

Here is exactly what you need to do:

### Step 1: Install PostgreSQL
PIDSMaker needs a database. Install PostgreSQL on your Linux VM:
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo service postgresql start
```
Set the password for the default `postgres` user (you'll need this in Step 4):
```bash
sudo -u postgres psql -c "ALTER USER postgres PASSWORD '12345678';"
```

### Step 2: Download & Restore the THEIA Dataset
PIDSMaker provides a script to download the pre-processed database dump.
```bash
# First download it (requires the Google Drive access token PIDSMaker provides)
./download_datasets.sh theia_e3 ya29.a0AT3oNZ8WJIin0xi014FqhzT_d-IfDfOyIFXeuaM1WqzPQmYVHWeoJpZQXNj_lTN5AnX7whGfcLZLy9bcdyBwRcpc7ljCXet5THOumdfBLV3DelK63Em8gmTIBKPt6N_7mcr2vUgeDuIXUyh7qiDTOnX0dpG22yP8n9PBdM0mLOXDLEc-Dglp0YexCxK5gpRChymsk1EaCgYKAbYSARcSFQHGX2MiHGBt-nmbKdIC_FAI9ohozw0206
# Restore the dump into your local PostgreSQL instance
pg_restore -U postgres -d postgres -1 data/theia_e3.dump
```

### Step 3: Install the Python Dependencies
Since you're not using their Docker container, you need to install PIDSMaker's code into your Python environment:
```bash
# Make sure you are in the PIDSMaker directory
pip install -e .
```
*(If you are missing things like PyTorch or DGL, you may need to install them manually based on their `pyproject.toml`)*

### Step 4: Tell PIDSMaker where your Database is
PIDSMaker looks for connection settings in your environment variables. Create a file called `.env.local` in the `PIDSMaker` directory (or just export these variables in your terminal) so it knows to connect to your local database instead of a Docker container:
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_USER=postgres
export DB_PASSWORD=your_password
```

### Step 5: Run the Baselines!
Now your VM has the database running and the Python code installed. You can run the exact commands from earlier:
```bash
python -m pidsmaker.main kairos THEIA_E3
python -m pidsmaker.main threatrace THEIA_E3
python -m pidsmaker.main magic THEIA_E3
```
Viewed compose-postgres.yml:1-34

This error means that the dataset was exported using **PostgreSQL 17**, but the default PostgreSQL installed on your Linux VM is an older version (like 12 or 14). Older versions of PostgreSQL cannot read database dumps created by newer versions.

Since you are not using Docker, you need to manually upgrade your Linux VM to PostgreSQL 17. 

Assuming you are on Ubuntu or Debian, run these exact commands in your terminal to install version 17:

### 1. Add the official PostgreSQL 17 repository
```bash
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'

wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo tee /etc/apt/trusted.gpg.d/pgdg.asc &>/dev/null

sudo apt update
```

### 2. Install PostgreSQL 17
First, remove the old version so it doesn't conflict:
```bash
sudo apt remove --purge postgresql postgresql-contrib -y
```
Now install version 17:
```bash
sudo apt install postgresql-17 postgresql-client-17 -y
```

### 3. Reset the Password and Restore
Just like before, reset the password so PIDSMaker can connect:
```bash
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'your_password';"
```

Now your `pg_restore` command will work perfectly! Try running it again:
```bash
pg_restore -U postgres -d postgres -1 data/theia_e3.dump
```

If you are still getting that error, it means your Linux VM is still using the old version of `pg_restore` that was left behind in your system's PATH, even though PostgreSQL 17 is installed.

To force Linux to use the brand new PostgreSQL 17 version of the tool, you just need to provide the **absolute path** to the executable. 

Run this command instead:
```bash
/usr/lib/postgresql/17/bin/pg_restore -U postgres -d postgres -1 data/theia_e3.dump
```

*(Note: If you get a "command not found" for that specific path, you can find exactly where version 17 is installed by running `find /usr -name pg_restore 2>/dev/null`, and then use the path that has "17" in it).*
