import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import javax.imageio.ImageIO;



public class Main_GMM {

    //3 arraylists pour enrg les tableaux de mu , cov et data
	public static ArrayList<double[]> mu = new ArrayList<>();
	public static ArrayList<double[][]>cov = new ArrayList<>();
	public static ArrayList<double[]> pixels = new ArrayList<>();
	public static int Threshold =30 ;
	//14
	
	public static  double[] Pi_k, Mu_k;
	public static  double[][]  cov_k;
	public static void main(String[] args) {
		
		try {
			File f=new File("zebra.jpg") ;
			BufferedImage originalImage = ImageIO.read(f);
			int k = 3;
			
			String output_image_name = f.getName().substring(0, f.getName().length() - 4) + "_output_k" + k
					+ f.getName().substring(f.getName().length() - 4, f.getName().length());
			//output image
			BufferedImage gmmJpg = Gmm(originalImage, k);
			ImageIO.write(gmmJpg, "jpg", new File(output_image_name));
		} catch (IOException e) {
			System.out.println(e.getMessage()+" test");
		}    
	}
	
	private static BufferedImage Gmm(BufferedImage originalImage, int k) {
		int w = originalImage.getWidth();
		int h = originalImage.getHeight();
		BufferedImage gmmImage = new BufferedImage(w, h, originalImage.getType());
		Graphics2D g = gmmImage.createGraphics();
		g.drawImage(originalImage, 0, 0, w, h, null);
		// Read rgb values from the image
		int[] rgb = new int[w * h];
		int count = 0;
		for (int i = 0; i < w; i++) {		
			for (int j = 0; j < h; j++) {
				rgb[count] = gmmImage.getRGB(i, j);
				count++;
				
			}
			
		}
		// Call EM algorithm to update the RGB values
		
   rgb= EM(rgb,k);
		// Write the new rgb values to the image
		count = 0;
		for (int i = 0; i < w; i++) {
			for (int j = 0; j < h; j++) {
				gmmImage.setRGB(i, j, rgb[count++]);
			}
		}
		return gmmImage;
	}
	
	
public static int[] EM(int [] rgb , int kk ){
	
	// get all rgb data put it in a table of a size of 3
	for (int i = 0; i < rgb.length; i++) {
			
			double[]data_x  = new double[3];
			
				data_x[0]=(double) getRed(rgb[i]) ;
				data_x[1]=(double) getGreen(rgb[i]) ;
				data_x[2]=(double) getBlue(rgb[i]) ;
				
			pixels.add(data_x);	
	}
	
	// new rgb table that we will return 
	int []newr = new int[pixels.size()];
	// assignement of every pixel to every cluster 
	int[] pixel_assignments = new int[pixels.size()];
		InitializeParameters3D(kk,rgb);	        
	//	double 	ChangeInLogLikelihood = 1;
		
		int ar=1;
		Double changeinlikelihood =Double.MAX_VALUE;
		while(ar<Threshold)
		{
			
	
			
			/*
			Double loglikelihood =0.0 ;
			for(int i=0 ; i<pixels.size();i++)
			{
				Double x=0.0 ;
				for(int j=0 ; j<kk;j++)
				{
					x=x+Pi_k[j]*GaussianFunction(pixels.get(i), mu.get(j), cov.get(j) );
				}
				
				loglikelihood = loglikelihood + log2(x);
			}
			System.out.println("loglikelihood "+loglikelihood); */
				// E- Step   Calculating Responsibilities...
			
		        Double[][] Rspb = new Double[pixels.size()][kk];        
				for(int i=0; i< pixels.size(); i++){
					double max_resp = 0;
					int cluster_index = 0;
					double denominator = 0;
					for(int l=0;l<kk;l++) denominator+=Pi_k[l]*GaussianFunction(pixels.get(i), mu.get(l), cov.get(l) );
					
					double s=0 ;
					for(int j=0;j<kk;j++) {
						Rspb[i][j]=0.0 ;
						double numerator =Pi_k[j]*GaussianFunction(pixels.get(i), mu.get(j), cov.get(j));
						Rspb[i][j]=numerator/denominator;
						//System.out.println(j+"  " +Rspb[i][j]);
						if(Rspb[i][j].isNaN()) {Rspb[i][j]=0.1;}
					//	s=s+Rspb[i][j];
						if (Rspb[i][j] >=max_resp) {
							max_resp = Rspb[i][j];
				// if the resp of a cluster is bigger than the previous max then change the cluster index
							cluster_index = j;
							//System.out.println(max_resp);
						}
						
					}
				// System.out.println(s+"            "+i);
					
					// Assign pixel to cluster
					pixel_assignments[i] = cluster_index;
				//	if(i==3)break;
					
				}
							
				// M-Step     Re-estimating Parameters...
				
				double[] N_k = new double[kk];
				
	            // calculating to total respo to use later
				for(int k=0; k<kk; k++){ 
					N_k[k]=0;// Calculating N_k's
					for(int n=0; n< pixels.size(); n++){
						N_k[k] = N_k[k] + Rspb[n][k];
					}
				
				}
				// Calculating new Pi_k's
				
				for(int k=0; k<kk; k++){
					Pi_k[k] = N_k[k]/pixels.size();
					
				}
			
			// Calculating new Mu_k's
				mu.clear();//clear the arraylist
				for(int k=0; k<kk; k++){ // for every cluster
					
					Mu_k  = new double[3];// create a new mu table
					
					for (int i=0;i<3;i++) { // 
						Double m =(double) 0;
						
					for(int n=0; n< pixels.size(); n++){
							
							m = m + Rspb[n][k]*pixels.get(n)[i];
							
							
						}
					
						
						Mu_k[i] = m/N_k[k] ;
					//	System.out.print(" mu["+i+"] = "+Mu_k[i]+" ");

						}
					//System.out.println("");
					
					mu.add(Mu_k);
				}
		/*	//	System.out.println("\n---------------mu after updates-------------------------------");
				for(int k=0; k<mu.size(); k++){
					//System.out.println();
					for(int n=0; n< mu.get(k).length; n++){
			//			System.out.print(" "+mu.get(k)[n]+" ");
					}
				}
				System.out.println("\n-----------------------------------------");*/
				/////////////////////////////////
				cov.clear();
				
				for(int k=0 ; k<kk;k++)
				{
					cov_k = new double[3][3];
					
					for (int i=0;i<3;i++)
					{
						
						for(int j=0;j<3;j++)
							{
							double c=0;
							for(int n=0; n< pixels.size(); n++){
			
								
								c=c + Rspb[n][k]*(((double)pixels.get(n)[i]-mu.get(k)[i])*((double)pixels.get(n)[j]-mu.get(k)[j]));
								
							}
							Double l =Math.sqrt(c/N_k[k]);  
							if(l.isNaN())
								if(i>0)
									l=cov_k[i-1][j];
							       else 
							    	   if(j>0) 
							    	   l=cov_k[1][j-1];
							    	   else
							    		   l=10.0;
							cov_k[i][j]=l;
							//System.out.print("cov_k["+i+"]["+j+"] = "+cov_k[i][j]+"  ");
						 
							}
						//System.out.println();
						
						
					}
					
				//	System.out.println();
					cov.add(cov_k);
				}
				
				
			
				
	        /*  
				for (int i=0;i<kk;i++) {
					System.out.println("pi: "+Pi_k[i]);
					System.out.println("mu: "+Mu_k[i]);
					System.out.println("sigma: "+cov_k[i]);
				}*/
				ar++;
			/*	
				Double newloglikelihood=0.0 ;
				
				for(int i=0 ; i<pixels.size();i++)
				{
					Double x=0.0 ;
					for(int j=0 ; j<kk;j++)
					{
						x=x+Pi_k[j]*GaussianFunction(pixels.get(i), mu.get(j), cov.get(j) );
					}
					
					newloglikelihood = newloglikelihood + log2(x);
				}
				changeinlikelihood=newloglikelihood-loglikelihood;
				System.out.println("change_loglikelihood "+changeinlikelihood);*/
				
		}// While	
		
		System.out.println("\n---------------pi -------------------------------");
		
		for(int k=0; k<kk; k++){
			System.out.print("Pi_k["+k+"] = "+Pi_k[k]+"\t");
			
		}
		
		System.out.println("\n-----------------------------------------");
		
		
		System.out.println("\n---------------mu -------------------------------");
		for(int k=0; k<mu.size(); k++){
			System.out.println();
			for(int n=0; n< mu.get(k).length; n++){
				System.out.print(" "+mu.get(k)[n]+" ");
			}
		}
		System.out.println("\n-----------------------------------------");
		System.out.println("-------------------------------cov------------------------------");
		for(int k=0 ; k<kk;k++)
		{
			
			for (int i=0;i<3;i++)
			{
				
				for(int j=0;j<3;j++)
					{
					System.out.print("cov_k["+i+"]["+j+"] = "+cov.get(k)[i][j]+"  ");
				 
					}
				System.out.println();
				
				
			}
			System.out.println();
			
		}
		
		System.out.println("-------------------------------done------------------------------");
		int [] colors = new int[kk];
		colors = getcolors(kk) ; //get num of colors equal to num of clusters
		for (int i = 0; i < pixels.size(); i++) {
			newr[i] = colors[pixel_assignments[i]]; // based on pixel assignments , return an rgb color for every pixel
		}
	return newr;
   } // function GaussianMixtureModel
	

   
	public static int[] getcolors(int k)
	{
		int[][] tab = new int[][]{{220,20,60},{124,252,0},{255,228,196},{32,178,170},{255,215,0},{70,130,180}};
		
		int[] values = new int[k];
		int e=10 ;
		Random ran = new Random();
		for (int i = 0; i < k; i++) {
			
			 
			//int rand_red = ran.nextInt(250);
		//	int rand_green = ran.nextInt(250);
		//	int rand_blue = ran.nextInt(250);
			
            int rand_red=tab[i][0];
            int rand_green = tab[i][1];
            int  rand_blue =tab[i][2];
         
			values[i] = /*((avg_alpha & 0x000000FF) << 24) | */ ((rand_red & 0x000000FF) << 16)
					| ((rand_green & 0x000000FF) << 8) | ((rand_blue & 0x000000FF) << 0);
			
		}
		return values ;
	}
	
	public static Double GaussianFunction(double [] input, double [] mu , double [][] cov ){     // return N(x_n|...)
		
	
		int n=input.length ;
		double det = determinant(cov) ;
	//	System.out.println("det= "+det);
		double[][] inv =  inverse(cov);
	   // System.out.println("input= "+input.getRowDimension()+" * "+input.getColumnDimension());
	   // System.out.println("mu= "+mu.getRowDimension()+" * "+mu.getColumnDimension());
		double[] x_mu = minus(input, mu) ;
		//double [] x_mu_tr =  transposeMatrix(x_mu);
		double[] m = mult2(inv, x_mu);
		
		double e = mult(x_mu, m);		
		Double Prob = 0.0;
		//Prob = Math.pow(Math.pow(2*3.14159265, n)*det ,-0.5)*Math.exp(-0.5*e);
		Double d = Math.sqrt(det) ;
		if (d.isNaN()) d=1.0;
		Prob = (1/(Math.pow(Math.sqrt(2*Math.PI), n)*d))*Math.exp(-0.5*e);
		//System.out.println(Math.sqrt(det));
		//if(Prob.isInfinite())Prob=0.0 ;
	 //   System.out.println(Prob);
	    return Prob;
	}

	
public static void InitializeParameters3D(int kk,int[] rgb){
	
	Pi_k  = new double[kk];        
	
		for (int i=0;i<kk;i++)Pi_k[i]=1./kk;   
		
		int [] init =kmeans(rgb, kk);
		for(int j=0 ; j<kk ; j++)
		{
			double r=0,g=0,b=0;
			r=(double) getRed(init[j]) ;
			g=(double) getGreen(init[j]) ;
			b=(double) getBlue(init[j]) ;
			
			Mu_k  = new double[3];
		
				Mu_k[0]=r ;
				Mu_k[1]=g ;
				Mu_k[2]=b ;
				
				
			
			mu.add(Mu_k);
			
		}
		System.out.println("\n\ninit mu");                  
		for(int k=0; k<mu.size(); k++){
			System.out.println();
			for(int n=0; n< mu.get(k).length; n++){
				System.out.print(mu.get(k)[n]+" ");
			}
		}
		System.out.println();
		for(int k=0 ; k<kk;k++)
		{
			cov_k = new double[3][3];
			
			for (int i=0;i<3;i++)
			{
				
				for(int j=0;j<3;j++)
					{
					double c=0;
					for(int n=0; n< pixels.size(); n++){
	
						
						c=c + (((double)pixels.get(n)[i]-mu.get(k)[i])*((double)pixels.get(n)[j]-mu.get(k)[j]));
						
					}
					Double l =Math.sqrt(c/pixels.size());  
					if(l.isNaN())
						if(i>0)
							l=cov_k[i-1][j];
					       else 
					    	   if(j>0) 
					    	   l=cov_k[1][j-1];
					    	   else
					    		   l=10.0;
					cov_k[i][j]=l;  
					//System.out.print("cov_k["+i+"]["+j+"] = "+cov_k[i][j]+"  ");
				 
					}
				//System.out.println();
				
				
			}
			
			System.out.println();
			cov.add(cov_k);
		}
		/*
		int t=10;
        int k=t;
		for(int o=0 ; o<kk; o++)
		{
			t=k;
			cov_k = new double[3][3];
			for (int i=0;i<3;i++)
			{
				
				for(int j=0;j<3;j++)
					{
				      if(i==1 && j==1 || i==2&&j==2)t=k;
					cov_k[i][j]=t;  
				 t=0;
					}
				
				
			}
		//	k+=35;
			cov.add(cov_k);
		}
		*/
		 	
		
	}

	
	// HELPER FUNCTIONS - to get individual R, G, and B values
			public static int getRed(int pix) {
				return (pix >> 16) & 0xFF;
			}

			public static int getGreen(int pix) {
				return (pix >> 8) & 0xFF;
			}

			public static int getBlue(int pix) {
				return pix & 0xFF;
			}

			/*public static int getAlpha(int pix) {
				return (pix >> 24) & 0xFF;
			}
			*/
			
			
			/*public static double log2(double num)
			{ if(num==0)
				return 0;
			  else 
			  {
				  return (Math.log(num)/Math.log(2));
			  }
			    
			}*/
			
			private static int[] kmeans(int[] rgb, int k) {
				// k = number of colors
				// Set k values initially to the colors of random pixels in the image.
				// Make sure they're all different from each other
				int[] k_values = new int[k];
				Random rand = new Random();
				for (int i = 0; i < k_values.length; i++) {
					int random_num;
					boolean contains_duplicate = true;
					if (i == 0) {
						random_num = rand.nextInt(rgb.length);
						k_values[i] = rgb[random_num];
					} else {
						do {
							random_num = rand.nextInt(rgb.length);
							for (int j = 0; j < i; j++) {
								if (j == i - 1 && k_values[j] != rgb[random_num]) {
									// If at the last element in k_values and
									// no duplicates detected
									k_values[i] = rgb[random_num];
									contains_duplicate = false;
								} else if (k_values[j] == rgb[random_num]) {
									// Exit for loop because duplicate color detected
									// and
									// try again with new random number
									j = i;
								}
							}
						} while (contains_duplicate);
					}
					System.out.println("Inital k mean " + i + ": " + k_values[i]);
				}

				// Group together similar pixels in the image
				// to corresponding cluster centers.
				// pixel assignments (by their index)
				int[] pixel_assignments = new int[rgb.length];
				int[] num_assignments = new int[k];

				// Cluster sums for current cluster values (represented by index)
			//	int[] alpha_sum = new int[k];
				int[] red_sum = new int[k];
				int[] green_sum = new int[k];
				int[] blue_sum = new int[k];

				// iterate until converged. Shouldn't take more than 100
				int max_iterations = 100;
				int num_iterations = 1;
				System.out.println("Clustering k = " + k + " points...");
				while (num_iterations <= max_iterations) {
					
					// Clear number of assignments list first
					for (int i = 0; i < k_values.length; i++) {
						num_assignments[i] = 0;
						//alpha_sum[i] = 0;
						red_sum[i] = 0;
						green_sum[i] = 0;
						blue_sum[i] = 0;
					}

					// Go through all pixels in rgb
					for (int i = 0; i < rgb.length; i++) {
						// Set min_dist initially to infinity (or very large number that
						// wouldn't appear as a distance anyways)
						double min_dist = Double.MAX_VALUE;
						int cluster_index = 0;
						// compare instance's RGB value to each cluster point
						for (int j = 0; j < k_values.length; j++) {
							//int a_dist = (getAlpha(rgb[i]) - getAlpha(k_values[j]));
							int r_dist = (getRed(rgb[i]) - getRed(k_values[j]));
							
							int g_dist = (getGreen(rgb[i]) - getGreen(k_values[j]));
							int b_dist = (getBlue(rgb[i]) - getBlue(k_values[j]));
							double dist = Math.sqrt( r_dist * r_dist + g_dist * g_dist + b_dist * b_dist);
							if (dist < min_dist) {
								min_dist = dist;
								cluster_index = j;
							}
						}
						// Assign pixel to cluster
						pixel_assignments[i] = cluster_index;
						num_assignments[cluster_index]++;
						// Add pixel's individual argb values to respective sums for use
						// later
						//alpha_sum[cluster_index] += getAlpha(rgb[i]);
						red_sum[cluster_index] += getRed(rgb[i]);
						green_sum[cluster_index] += getGreen(rgb[i]);
						blue_sum[cluster_index] += getBlue(rgb[i]);
					}

					// update previous assignments list
					for (int i = 0; i < k_values.length; i++) {
						//int avg_alpha = (int) ((double) alpha_sum[i] / (double) num_assignments[i]);
						int avg_red = (int) ((double) red_sum[i] / (double) num_assignments[i]);
						int avg_green = (int) ((double) green_sum[i] / (double) num_assignments[i]);
						int avg_blue = (int) ((double) blue_sum[i] / (double) num_assignments[i]);

						k_values[i] =  ((avg_red & 0x000000FF) << 16)
								| ((avg_green & 0x000000FF) << 8) | ((avg_blue & 0x000000FF) << 0);
					}
					num_iterations++;
				}

				
				System.out.println("Clustering image converged.");
				for (int i = 0; i < k_values.length; i++) {
					System.out.println("Final k mean " + i + ": " + k_values[i]);
				}
				return k_values;
			}
			
			 private static double determinant(double[][] matrix) {
			        if (matrix.length != matrix[0].length)
			            throw new IllegalStateException("invalid dimensions");

			        if (matrix.length == 2)
			            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];

			        double det = 0;
			        for (int i = 0; i < matrix[0].length; i++)
			            det += Math.pow(-1, i) * matrix[0][i]
			                    * determinant(minor(matrix, 0, i));
			        return det;
			    }

			    private static double[][] inverse(double[][] matrix) {
			        double[][] inverse = new double[matrix.length][matrix.length];

			        // minors and cofactors
			        for (int i = 0; i < matrix.length; i++)
			            for (int j = 0; j < matrix[i].length; j++)
			                inverse[i][j] = Math.pow(-1, i + j)
			                        * determinant(minor(matrix, i, j));

			        // adjugate and determinant
			        double det = 1.0 / determinant(matrix);
			        for (int i = 0; i < inverse.length; i++) {
			            for (int j = 0; j <= i; j++) {
			                double temp = inverse[i][j];
			                inverse[i][j] = inverse[j][i] * det;
			                inverse[j][i] = temp * det;
			            }
			        }

			        return inverse;
			    }
			    public static double[][] transposeMatrix(double [][] m){
			        double[][] temp = new double[m[0].length][m.length];
			        for (int i = 0; i < m.length; i++)
			            for (int j = 0; j < m[0].length; j++)
			                temp[j][i] = m[i][j];
			        return temp;
			    }

			    private static double[][] minor(double[][] matrix, int row, int column) {
			        double[][] minor = new double[matrix.length - 1][matrix.length - 1];

			        for (int i = 0; i < matrix.length; i++)
			            for (int j = 0; i != row && j < matrix[i].length; j++)
			                if (j != column)
			                    minor[i < row ? i : i - 1][j < column ? j : j - 1] = matrix[i][j];
			        return minor;
			    }
			    
			    public static double [] minus(double [] a , double [] b)
			    {
			    	double [] c = new double [a.length];
			    	for(int i=0; i<a.length; i++)
			    	   {
			    	       c[i] = a[i] - b[i];
			    	    }
			    	return c ;
			    } 
			    public static double [] mult2 (double [][] a , double [] b)
				{
					double [] c = new double[b.length];
					
					
					for (int i=0 ; i<a[0].length;i++)
					{double m =0 ;
						for (int j=0 ; j<b.length ; j++)
						{
							m=m+(b[j]*a[i][j]);
						}
						c[i]=m;
					}
							
					return c ;		
					
				}
			    
			    public static double  mult (double [] a , double [] b)
				{
			    	double m =0 ;
					
					
					
					
						for (int j=0 ; j<b.length ; j++)
						{
							
							m=m+(b[j]*a[j]);
							
						}
						
					
							
					return m ;		
					
				}
			    
			    public static double log2(double num)
				{ if(num==0)
					return 0;
				  else 
				    return (Math.log(num)/Math.log(2));
				}

}
