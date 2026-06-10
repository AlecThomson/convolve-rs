fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stub = convolve_rs::stub_info()?;
    stub.generate()?;
    Ok(())
}
