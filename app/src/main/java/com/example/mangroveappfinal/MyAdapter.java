package com.example.mangroveappfinal;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.text.BreakIterator;
import java.util.ArrayList;

public class MyAdapter extends RecyclerView.Adapter<MyAdapter.MyViewHolder>{

    ArrayList<Model> mList;
    Context context;

    public MyAdapter(Context context, ArrayList<Model> mlist){

        this.mList = mlist;
        this.context = context;

    }

    @NonNull
    @Override
    public MyViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View v = LayoutInflater.from(context).inflate(R.layout.item, parent, false);
        return new MyViewHolder(v);

    }

    @Override
    public void onBindViewHolder(@NonNull MyViewHolder holder, int position) {

        Model model = mList.get(position);
        holder.Bakhaw.setText(model.getBakhaw());
        holder.Pagatpat.setText(model.getPagatpat());
        holder.Bungalon.setText(model.getBungalon());






    }

    @Override
    public int getItemCount() {
        return mList.size();
    }

    public static class MyViewHolder extends RecyclerView.ViewHolder{


        TextView Bakhaw,Pagatpat,Bungalon;


        public MyViewHolder(@NonNull View itemView) {
            super(itemView);


            Bakhaw = itemView.findViewById(R.id.bakhaw_text);
            Pagatpat = itemView.findViewById(R.id.pagatpat_text);
            Bungalon = itemView.findViewById(R.id.bungalon_text);




        }
    }

}
