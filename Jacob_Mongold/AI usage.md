# AI Usage

I'd never used AWS before this project. The data was sitting in a private S3 bucket (naccmri-quickaccess-sub) and all I really had was a credentials file someone sent me and a rough idea that I needed to download MRIs. So I used Claude a lot to figure out how to actually get the data onto my machine. Later on I also used it to dig out from a really nasty bug between pandas and Weights & Biases.

## What I asked it for help with

### Setting up AWS credentials
The first thing was just credentials. I didn't know boto3 looked at ~/.aws/credentials by default, or that you needed to set a region. Claude explained the default profile setup and gave me a quick one-liner (boto3.client("s3").list_buckets()) so I could confirm the connection actually worked before writing any real code.

### Exploring the bucket
After that I had to figure out what was even in the bucket. I asked how to list objects without pulling everything down, and that's where I learned about list_objects_v2 and paginators. Claude wrote a small loop that walked the prefixes (investigator/MRI/within1yr/nifti/, investigator/MRI/all/nifti/) so I could see the layout. I had been picturing S3 like a folder tree on my laptop and that's just not how it works, which took a minute to wrap my head around.

### Matching S3 files to my cohort
The next part was matching the S3 files to the patients in my cohort CSV. The MRI zips were named like NACC123456_something.zip. I asked Claude how to pull the IDs out and it suggested a regex (re.compile(r"NACC(\d{6})")) and building a dict from NACCID to its list of keys. Then I could intersect that with my cohort to see who I could actually get imaging for. That ended up being most of scripts/s3_find_mri.py.

### Safe downloads and resuming
Downloading was where I would've messed things up the most on my own. I wanted to grab a zip, unzip it into a per-patient folder, and not leave temp files lying around if something crashed. Claude showed me the try/finally pattern with zip_path.unlink(missing_ok=True), and also told me to add a check that skips patients I'd already downloaded. That second part saved me later when my connection dropped halfway through a long run and I could just re-run the script.

### Parallelizing downloads
I also asked about parallelizing the downloads because sequential was way too slow. Claude pointed me at ThreadPoolExecutor from concurrent.futures and explained why threads work fine here even though Python has the GIL, because the bottleneck is network I/O, not CPU.

### Resampling the cohort
Once I had the list of patients S3 actually had, I needed to re-filter my cohort and rebuild the manifest so I wasn't training on people with no imaging. Claude helped me wire that into scripts/resample_from_s3.py.

### The pandas + Weights & Biases bug
I had W&B set up to log training metrics and it was working fine until I started logging some pandas DataFrames (per-class accuracy stuff, confusion matrix counts). I'd get cryptic errors coming out of wandb's internal code, sometimes about dtypes, sometimes about serialization, sometimes the run would just hang and never sync. Different errors on different runs even when nothing about my code had changed.

I spent a long time going back and forth with Claude trying to narrow it down. We tried converting the DataFrames to plain dicts before logging, casting columns to native python types, downgrading wandb, downgrading pandas, wrapping everything in wandb.Table, none of it stuck. Some fixes worked for one run and broke the next. At some point Claude basically said the version interaction between the pandas I was using and the wandb client was just broken, and that for a project on a deadline it wasn't worth fighting.

So I ripped W&B out. I replaced it with logging metrics to local CSVs and making the plots myself with matplotlib.

## What I learned

A lot of stuff that I don't think I would've picked up just by reading docs.

### S3 is not a filesystem
S3 isn't really a filesystem, it's a flat key/value store and the slashes in keys are just part of the name. Once that clicked, things like paginators and prefixes started making more sense. Speaking of paginators, my first instinct was to just call list_objects_v2 once and read the result. Turns out it caps at 1000 keys per response, so without paging I would have silently missed most of the bucket and not even known.

### Threads aren't always useless
I also got a real answer to the "are Python threads useless?" question that I'd seen thrown around online. They're useless for CPU-bound stuff because of the GIL, but for I/O like S3 downloads they're great. That was one of those things I'd kind of memorized as a fact but didn't really get until I saw it work.

### Knowing when to stop debugging
The W&B thing taught me something different. I sunk way too many hours into trying to fix that bug because I wanted the nice dashboard. Eventually the right call was just to delete the dependency and move on.