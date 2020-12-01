use std::fs::read_to_string;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<std::error::Error>> {
    let p :Vec<u8> = read_to_string("data/manual.txt").unwrap()[0..100_000_000].bytes().collect::<Vec<u8>>();
    println!("aa{}", p.len());
    for i in 0..100 {
        let filename = format!("data/txt/man{}.txt", i);
        let mut file = File::create(filename)?;
        println!("{:?}",&p[0+100000*i .. (i+1)*100000]);
        file.write_all(&p[0+100000*i .. (i+1)*100000])?;
        file.flush()?;
    }
    Ok(())
}
