import subprocess
import time
import pandas as pd
import argparse
from io import StringIO
from Bio import SeqIO
import os

doc = """This script can be imported or run from the command line.
It takes a fasta file and a diamond database and runs diamond blastp.
It can either write the output to a file or return a pandas dataframe.

Example usage:
    python run_diamond.py query.fasta db.fasta output.tsv --add_seq --add_self
"""

file_path = os.path.dirname(os.path.realpath(__file__))

def get_current_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def record_time(func):
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        start_time = time.time()
        print("Start time for %s: %s" % (func_name, get_current_time()))
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = round(end_time - start_time, 2)
        print("End time for %s: %s" % (func_name, get_current_time()))
        print("Total time for %s: %s seconds" % (func_name, total_time))
        print('----------------------------------------------------------------\n')
        return result
    return wrapper

@record_time
def run_diamond(
    fasta_file,
    database,
    diamond_path:str=os.path.join(file_path, "diamond"),
    evalue:float=10,
    sensitivity:str="--ultra-sensitive",
    threads:int=None,# if not specified, use diamond default
    add_seq:bool=False,
    output_file:str=None,
    return_df:bool=False,
    processed_file:str=None,
)->pd.DataFrame:
    """
    Run diamond blastp with the given fasta file and database.
    If output file is given, write to it.
    If not, return a pandas dataframe.

    Args:
        fasta_file: fasta file
        database: blast database
        blastp_path: path to blastp, default is blastp
        output_file: output file
        evalue: evalue cutoff, default is 0.001
        threads: number of threads, default is 8
        add_seq: add sequence to the output, default is False
                note, if add seq, a fai file is required on the same dir as the database
                if fai file is not found, it will be created, but samtools is required
        return_df: return a pandas dataframe, default is False
    """
    check_diamond(diamond_path)
    # output format
    print(f"searching {fasta_file} against {database}")
    outfmt = "6 qseqid sseqid bitscore pident evalue qlen slen nident"
    columns = ["query", "target", "bitscore", "pident" , "evalue", "qlen", "slen", "nident"]
    if add_seq:
        outfmt += " full_sseq"
        columns.append("target_seq")

    cmd = [
        diamond_path, "blastp",
        "--db", database,
        "--query", fasta_file,
        "--out", output_file,
        "--max-target-seqs", "500",
        "--quiet"
    ]
    cmd.extend(["--outfmt"])
    cmd.extend(outfmt.split())

    if evalue is not None:
        cmd.extend(["--evalue", str(evalue)])

    if threads is not None:
        cmd.extend(["--threads", str(threads)])

    if sensitivity:            
        cmd.extend([sensitivity])
    

    # run diamond
    P = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = P.communicate()
    if P.returncode != 0:
        raise RuntimeError("Diamond failed: %s" % err.decode())

    # read in pandas dataframe
    columns = ["query", "target", "bitscore", "pident" , "evalue", "qlen", "slen", "nident"]
    if output_file is not None:
        if processed_file is not None:
            # if unfinished file is given, cat the current output file with the unfinished file
            lines = []
            with open(processed_file, "r") as f:
                lines.extend(f.readlines())
            with open(output_file, "r") as f:
                lines.extend(f.readlines())
            with open(output_file, "w") as f:
                f.writelines(lines)
        df = pd.read_csv(output_file, sep="\t", names=columns)
    else:
        tsv_out = StringIO(out.decode("utf-8"))
        df = pd.read_csv(tsv_out, sep="\t", names=columns)
    

    df.target = df.target.apply(lambda x: x.split("|")[1] if "|" in x else x)
    
    if output_file is not None:
        # if output file is given, write to it
        df.to_csv(output_file, sep="\t", index=False)
    if return_df:
        # if return_df is True, return a pandas dataframe
        return df

def check_diamond(diamond_path:str="diamond"):
    """
    Check if diamond is installed.
    """
    cmd = [diamond_path, "--version"]
    P = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = P.communicate()
    if P.returncode != 0:
        raise Exception("Diamond failed with error: %s" % err)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", help="fasta file")
    parser.add_argument("database", help="diamond database")
    parser.add_argument("output_file", help="output file")
    parser.add_argument("--diamond_path", help="path to diamond, default is diamond", default="diamond")
    parser.add_argument("-e", "--evalue", help="evalue cutoff, default is 0.001", type=float, default=None)
    parser.add_argument("-s", "--sensitivity", help="sensitivity, default is --ultra-sensitive", default="--ultra-sensitive")
    parser.add_argument("-n", "--threads", help="number of threads, default is 8", type=int, default=None)
    parser.add_argument("--add_seq", help="add sequence to the output, default is False", action="store_true")
    args = parser.parse_args()
    run_diamond(
        args.query,
        args.database,
        args.output_file,
        diamond_path=args.diamond_path,
        evalue=args.evalue,
        sensitivity=args.sensitivity,
        threads=args.threads,
        add_seq=args.add_seq,
    )