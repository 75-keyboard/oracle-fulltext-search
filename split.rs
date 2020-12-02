use std::fs::read_to_string;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<std::error::Error>> {
    let p :Vec<u8> = read_to_string("data/manual.txt").unwrap().bytes().collect::<Vec<u8>>();
    println!("aa{}", p.len());
    let h = 100;
    for i in 0..h {
        let filename = format!("data/txt/man{}.txt", i);
        let mut file = File::create(filename)?;
        let l = p.len() / h;
        file.write_all(&p[0+l*i .. (i+1)*l])?;
        file.flush()?;
    }
    Ok(())
}
